"""
Calculate junction angle of PDB structure (a single model single chain)
If a PDB have multi chain and multi model then we have to re-write the PDB before using the code

@author: Pradeep Kumar Yadav and Pratiti Bhadra


@usage: python3 junction_angle.py xxxx.pdb  (xxxx.pdb is the input)
        output: xxxx_JA.csv


@ DSSP execulable file is different for different Operating System
dssp for Linux 
dssp-3.0.0-win32.exe for Windows

"# DSSP execulatable" should be changed as per Operating System

"""




from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
import sys
import os
import pandas as pd
import numpy as np


# function for calculation of angle using three points
def angle_pt(pt1,pt2,pt3,pt4):


    vector_1 = pt2 - pt1
    vector_2 = pt4 - pt3


    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)

    return(round(np.degrees(angle),1))


# take .pdb file a argument from command line
#PDB = sys.argv[1]
path ='C:/Users/Pradeep Kumar Yadav/Downloads/20211221_181717_pratiti_bhadra_(pratiti.bhadra@pharmcadd.com)_Explore_and_Understand_PDB_str/'
p = PDBParser()
structure = p.get_structure("1MOT", path + "1a12.pdb")
# read pdb file using Biopython for dssp
#p = PDBParser()
#structure = p.get_structure("prot",PDB)
model = structure[0]


# extract coor of CA atoms
CA_coord = []

for model in structure.get_list():
    for chain in model.get_list():
        for residue in chain.get_list():
                if residue.has_id("CA"):
                        CA_coord.append(residue['CA'].get_coord())


df_CA_coord = pd.DataFrame(CA_coord,columns=['xcoord','ycoord','zcoord'])


# DSSP for secondary structure
#dssp_data = pd.DataFrame(DSSP(model,PDB,dssp='./dssp'))  # DSSP execulatable dssp for Linux OR dssp-3.0.0-win32.exe for Windows
dssp = DSSP(model, "1a12.pdb", dssp= path + 'dssp-3.0.0-win32.exe')
dssp_data=pd.DataFrame(dssp)
dssp_columns = ["residue_no", "Amino acid", "Secondary structure","Relative ASA","Phi","Psi","NH–>O_1_relidx","NH–>O_1_energy",
              "O–>NH_1_relidx","O–>NH_1_energy","NH–>O_2_relidx","NH–>O_2_energy","O–>NH_2_relidx","O–>NH_2_energy"]

dssp_data.columns = dssp_columns 


# DataFrame of residue no and SS
ResNo_SS =  dssp_data[["residue_no","Secondary structure"]]

# combine ResNo_SS and df_CA_coord
# dataframe (residueno,SS,CA_xcoord,CA_ycoord,CA_zcoord)
residue_data= pd.concat((ResNo_SS, df_CA_coord),axis=1)


# Make 'T'(turn), 'S'(bend) and '-'(coil) secondary structure as 'C'(coild)
residue_data_HBC = residue_data.replace(['T','S','-'], 'C')

# drop those rows where continuation of secondary structure does not noticed 
# like HHHBCCC , then drop the residue corresponding to B
list_of_drop_singleSS = []
for i in range(1,len(residue_data_HBC)-1):
    if residue_data_HBC.iloc[i]['Secondary structure'] != residue_data_HBC.iloc[i-1]['Secondary structure']:
        if residue_data_HBC.iloc[i]['Secondary structure'] != residue_data_HBC.iloc[i+1]['Secondary structure']:
                list_of_drop_singleSS.append(i)


continuousSS_residue_data_HBC = residue_data_HBC.drop(labels=list_of_drop_singleSS, axis=0)
continuousSS_residue_data_HBC = continuousSS_residue_data_HBC.reset_index() # reset index of dataframe after droping row

# The SS_axis vector of continuous secondary strcuture (SS)
# store CA of start and end point of SS
list_of_drop = [0]
for i in range(1,len(continuousSS_residue_data_HBC)-1):
    if continuousSS_residue_data_HBC.iloc[i]['Secondary structure'] == continuousSS_residue_data_HBC.iloc[i-1]['Secondary structure']:
        if continuousSS_residue_data_HBC.iloc[i]['Secondary structure'] == continuousSS_residue_data_HBC.iloc[i+1]['Secondary structure']:
                list_of_drop.append(i)

SS_vector_data = continuousSS_residue_data_HBC.drop(labels=list_of_drop, axis=0)
SS_vector_data = SS_vector_data.reset_index()
    
 
# Calculate the angle
# 12EEEECCCCC20 , EE vector 12E-15E and CC vector 16C-20C. Then angle between these two vector
# output 12,20,E,C,angle
cnt = 0
for i in range(0,len(SS_vector_data)-2,2):
    CA_1 = np.asarray([SS_vector_data.iloc[i]['xcoord'],SS_vector_data.iloc[cnt]['ycoord'],SS_vector_data.iloc[i]['zcoord']])
    CA_2 = np.asarray([SS_vector_data.iloc[i+1]['xcoord'],SS_vector_data.iloc[cnt+1]['ycoord'],SS_vector_data.iloc[i+1]['zcoord']])
    CA_3 = np.asarray([SS_vector_data.iloc[i+2]['xcoord'],SS_vector_data.iloc[cnt+2]['ycoord'],SS_vector_data.iloc[i+2]['zcoord']])
    CA_4 = np.asarray([SS_vector_data.iloc[i+3]['xcoord'],SS_vector_data.iloc[cnt+3]['ycoord'],SS_vector_data.iloc[i+3]['zcoord']])
    

    junction_angle = angle_pt(CA_1, CA_2, CA_3, CA_4)
    
    data = [[SS_vector_data.iloc[i]['residue_no'],SS_vector_data.iloc[i+3]['residue_no'],SS_vector_data.iloc[i]['Secondary structure']+SS_vector_data.iloc[i+3]['Secondary structure'],junction_angle]]

    if i == 0:
        output=pd.DataFrame(data)
    else:
        output=output.append(data,ignore_index = True)
        



op_columns = ['Start_Residue','End_Residue','SS','junction angle']
output.columns = op_columns


# write residue_no, SS and junction_angle in .csv file
output_file = PDB[0:4] + '.csv'
output.to_csv(output_file)

