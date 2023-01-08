# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 11:46:59 2022

@author: Pradeep Kumar Yadav
"""
import pandas as pd
import numpy as np

#reading and writing the input file"
mole_data= pd.read_csv('junction.csv')
# test = mole_data.iloc[:,1]
# for i in range(len(test)):
#     if test[i]==test[i+1]:
#         continue
#     else:
#         break

sec_structure= {'H':1,'B':2,'E':3,'G':4,'I':5,'T':6,'S':6,'-':6}

sec_score=mole_data.iloc[:,1].replace(sec_structure)
mole_data_temp = mole_data.copy()
mole_data_temp['sec_score'] = sec_score
#mole_data_temp['flag'] = mole_data_temp['sec_score'].diff()

# for i in range(1,7):
#       temp=[]
#       print( mole_data_temp[mole_data_temp['sec_score']==i])
#       temp=mole_data_temp[mole_data_temp['sec_score']==i]
#       # temp['diff']=mole_data_temp['residue_number'].diff()
#       # print('\n')



ct=1
cts=[]
flag=[]
for i in range(len(mole_data_temp)):
    
    if i<len(mole_data_temp)-1:
        if (mole_data_temp.iloc[i]['sec_score'] == mole_data_temp.iloc[i+1]['sec_score']): 
            cts.append(ct)
            ct=ct+1
        else :
              cts.append(ct)
              ct=1
    if i==len(mole_data_temp)-1: 
        if (mole_data_temp.iloc[i-1]['sec_score'] == mole_data_temp.iloc[i]['sec_score']): 
            cts.append(ct)
            ct=ct+1
        else :
              cts.append(1)

mole_data_temp['rep_num']= cts
#mole_data_temp['diff']=mole_data_temp['sec_score'].diff()

#==============================================================================

flags1=[0]*76
flags2=[0]*76
for i in range(len(mole_data_temp)):
    
    if i<len(mole_data_temp)-1: 
          if  mole_data_temp.iloc[i]['rep_num']>= mole_data_temp.iloc[i+1]['rep_num']:
                flags1[i]=0
                flags1[i+1]=1
          else:
              if  mole_data_temp.iloc[i]['rep_num']> mole_data_temp.iloc[i+1]['rep_num']:
                flags1[i]=1
    
    if i==len(mole_data_temp)-1: 
        if  mole_data_temp.iloc[i-1]['rep_num']<mole_data_temp.iloc[i]['rep_num']:
                flags1[i]=1
        else:
                flags1[i]=0

#mole_data_temp['flags1']=  flags1

flags2=[0]*76
for i in range(len(mole_data_temp)):
         
          if i<len(mole_data_temp)-1: 
                if  mole_data_temp.iloc[i]['rep_num']> mole_data_temp.iloc[i+1]['rep_num']:
                    flags2[i]=1
          else:
                flags2[i]=0
                
      
#mole_data_temp['flags2']=  flags2   
mole_data_temp['selected']=np.array(flags1)+np.array(flags2)      


finaldf=mole_data_temp[mole_data_temp['selected']==1]
#==============================================================================
      

# sel=[False]*76
# j=1
# for i in range(len(mole_data_temp)-1):
#     if  mole_data_temp.iloc[i]['rep_num']< mole_data_temp.iloc[i+1]['rep_num'] and mole_data_temp.iloc[i]['rep_num']< mole_data_temp.iloc[i+1]['rep_num'] :
#         sel[i]=True
#     else:
#         sel[i]=False
 
#     # if mole_data_temp.iloc[i]['flag']==0:
#     #     sel[i]=True
# mole_data_temp['selected']=sel

# final_df=mole_data_temp[mole_data_temp['selected']==True]

#C1=len(mole_data_temp)

# for i in range(3):
#     C2=1
#     while mole_data_temp.iloc[i]['sec_score'] == mole_data_temp.iloc[i+1]['sec_score']:
#         #if  len(mole_data_temp.iloc[i]['sec_score'])>76:
#             #break
#         #print(C2)
#         C2 = C2+1
        
        
#     # if C2==1:
#     #     C2=1
        
#     if (C2>1):
#         final_atom.append(mole_data_temp.iloc[i])
#         final_atom.append(mole_data_temp.iloc[i + C2-1])
        
# i=i + C2
        

        
    # if ( mole_data_temp.iloc[i]['sec_score'] == mole_data_temp.iloc[i+1]['sec_score']):
        
    #     final_atom.append(mole_data_temp.iloc[i])
    #     print()
        
    #     if(mole_data_temp.iloc[i+1]['sec_score'] == mole_data_temp.iloc[i+1]['sec_score'] == mole_data_temp.iloc[i+2]['sec_score']):
    #         print(True)
    # else:
    #     print(False)
        



# def epoch(x):
#     for i in range(len(mole_data)):
#         if sec_structure[0] == mole_data[i]:
#             continue
#         else:
#             break
            
                
        # splitting at ', ' into Data frame
#new = mole_data_temp["sec_score"].str.partition(False)
  
# making separate first atom column from new data frame
#mole_data_temp["sec_score"]= new[0]
   
# making separate last atom column from new data frame
#mole_data_temp["sec_score"]= new[-1]
   
# Dropping old Name columns
#mole_data_temp.drop(columns =["sec_score"], inplace = True)
   
# df display
#mole_data_temp  
        



#import pandasql
#sub_data=pandasql.sqldf("SELECT ROW_NUMBER() OVER (PARTITION BY sec_score,residue_number order by residue_number) AS minrn, ROW_NUMBER() OVER (PARTITION BY sec_score,residue_number order by residue_number desc) AS maxrn FROM mole_data_temp ")

