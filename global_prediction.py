# -*- coding: utf-8 -*-
from tensorflow.keras.layers import (BatchNormalization, Bidirectional,
                                     Dropout, Dense, LSTM)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import openpyxl
import random
import pickle
import joblib
import bz2
import os


# Set dictionary for "data" matrix column indexing
COL = {
    'index': 0, 'pdb': 1, 'chain': 2, 'number': 3, 'residue': 4, 'A': 5,
    'R': 6, 'N': 7, 'D': 8, 'C': 9, 'Q': 10, 'E': 11, 'G': 12, 'H': 13,
    'I': 14, 'L': 15, 'K': 16, 'M': 17, 'F': 18, 'P': 19, 'S': 20, 'T': 21,
    'W': 22, 'Y': 23, 'V': 24, 'ss8_B': 25, 'ss8_E': 26, 'ss8_G': 27,
    'ss8_H': 28, 'ss8_I': 29, 'ss8_C': 30, 'ss8_S': 31, 'ss8_T': 32,
    'ss3_C': 33, 'ss3_E': 34, 'ss3_H': 35, 'Steric Param': 36, 'Polarity': 37,
    'Volume': 38, 'Hydrophobicity': 39, 'Isoelectric Pt': 40, 'Helix Prob': 41,
    'Sheet Prob': 42, 'SASA': 43, 'phi': 44, 'psi': 45, 'omega': 46,
    'theta': 47, 'tau': 48, 'CACN': 49, 'CNCA': 50, 'NCAC': 51, 'D:C-N': 52,
    'D:N-CA': 53, 'D:CA-C': 54,
    # Function "data_augmentation" generated columns
    # sine / cosine of target dihedral angles
    'sin(phi)': 55, 'cos(phi)': 56, 'sin(psi)': 57, 'cos(psi)': 58,
    'sin(theta)': 59, 'cos(theta)': 60, 'sin(tau)': 61, 'cos(tau)': 62,
    # Position flag class variables   # need to understand in depth
    'position_flag': 63, 'pfc_start': 64, 'pfc_middle': 65, 'pfc_end': 66,
    # One Hot Encoded residues
    'aa_A': 67, 'aa_C': 68, 'aa_D': 69, 'aa_E': 70, 'aa_F': 71, 'aa_G': 72,
    'aa_H': 73, 'aa_I': 74, 'aa_K': 75, 'aa_L': 76, 'aa_M': 77, 'aa_N': 78,
    'aa_P': 79, 'aa_Q': 80, 'aa_R': 81, 'aa_S': 82, 'aa_T': 83, 'aa_V': 84,
    'aa_W': 85, 'aa_Y': 86
}

# Dict to map 1 letter res codes to respective int values for One Hot Encoding
AA_MAP = {
    'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8,
    'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16,
    'V': 17, 'W': 18, 'Y': 19
}


def data_augmentation(data):
    """
    Reads in and modifies data loaded from a prespecified file as numpy array.
    This function will contain any feature engineering steps desired.

    Args:
        data: numpy array with predefined protein structure prediction data.

    Returns:
        numpy array: returns engineered data array with updated information for
            structure prediction.

    NOTE: Any column added to the array for feature engineering will have
        to be added to the relevant column list in the "split_data" function.
    """

    # Fill NA values for neural network training
    data[data == 'nan'] = 0

    # Set sine and cosine of dihedral angles for more accurate predictions
    data = np.hstack((
        data,
        np.sin(data[:, COL['phi']].astype(float) *
               np.pi / 180).reshape(-1, 1),
        np.cos(data[:, COL['phi']].astype(float) *
               np.pi / 180).reshape(-1, 1),
        np.sin(data[:, COL['psi']].astype(float) *
               np.pi / 180).reshape(-1, 1),
        np.cos(data[:, COL['psi']].astype(float) *
               np.pi / 180).reshape(-1, 1),
        np.sin(data[:, COL['theta']].astype(float) *
               np.pi / 180).reshape(-1, 1),
        np.cos(data[:, COL['theta']].astype(float) *
               np.pi / 180).reshape(-1, 1),
        np.sin(data[:, COL['tau']].astype(float) *
               np.pi / 180).reshape(-1, 1),
        np.cos(data[:, COL['tau']].astype(float) *
               np.pi / 180).reshape(-1, 1),
    ))
    #print(data)
    # Designate position flag class: 0 at start, 2 at end, 1 in between
    conditions = [data[:, COL['phi']] == '0', data[:, COL['psi']] == '0']
    classes = [0, 2] # not able to understand
    #there should be some some proper feature understanding at this point
    data = np.hstack((
        data,
        np.select(conditions, classes, default=1).reshape(-1, 1)
    ))

    # Create and append One Hot Encoder columns for position flag
    pfc = data[:, -1].astype(int)
    ohef_matrix = np.zeros((pfc.size, pfc.max() + 1))
    ohef_matrix[np.arange(pfc.size), pfc] = 1
    data = np.hstack((
        data,
        ohef_matrix
    ))

    # Create and append One Hot Encoder columns for residues
    aa = np.array([AA_MAP[a] for a in data[:, COL['residue']]]).astype(int)
    ohef_matrix = np.zeros((aa.size, aa.max()+1))
    ohef_matrix[np.arange(aa.size), aa] = 1
    data = np.hstack((
        data,
        ohef_matrix
    ))

    return data


def split_data(data):
    """
    Takes engineered 'data' matrix and creates training and testing variables
    for model training and evaluation.

    Args:
        data: numpy array with predefined protein structure prediction data.

    Returns:
        dictionary: returns dictionary with relevant variables for model
            training and evaluation.
    """

    # If you would like to evaluate the model on a chain within the 'data' var,
    # Extract the chain in order to attempt to predict on unseen data
    # data = data[~((data[:, 1] == '1YQ3') & (data[:, 2] == 'C'))]

    # Array columns to be used as features for LSTM model
    window_cols = [
        'index', 'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K',
        'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'ss8_B', 'ss8_E', 'ss8_G',
        'ss8_H', 'ss8_I', 'ss8_C', 'ss8_S', 'ss8_T', 'ss3_C', 'ss3_E', 'ss3_H',
        'Steric Param', 'Polarity', 'Volume', 'Hydrophobicity',
        'Isoelectric Pt', 'Helix Prob', 'Sheet Prob', 'SASA', 'pfc_start',
        'pfc_middle', 'pfc_end', 'aa_A', 'aa_C', 'aa_D', 'aa_E', 'aa_F',
        'aa_G', 'aa_H', 'aa_I', 'aa_K', 'aa_L', 'aa_M', 'aa_N', 'aa_P',
        'aa_Q', 'aa_R', 'aa_S', 'aa_T', 'aa_V', 'aa_W', 'aa_Y',
        'position_flag'
    ]

    # Array columns to be used as targets for LSTM model
    y_cols = [
        'sin(phi)', 'cos(phi)', 'sin(psi)', 'cos(psi)', 'sin(theta)',
        'cos(theta)', 'sin(tau)', 'cos(tau)', 'CACN', 'CNCA', 'NCAC',
        'D:C-N', 'D:N-CA', 'D:CA-C'
    ]

    # Reform data to take only desired columns
    all_cols = [COL[name] for name in window_cols + y_cols]
    data = data[:, all_cols].astype(float)
    #print(data)
    # Scale PSSM values of whole population
    pssm_scaler = MinMaxScaler(feature_range=(-1, 1))
    data[:, 1:21] = pssm_scaler.fit_transform(data[:, 1:21])

    # Scale physical property values of whole population
    pp_scaler = MinMaxScaler(feature_range=(-1, 1))
    data[:, 32:40] = pp_scaler.fit_transform(data[:, 32:40])

    # Randomly generate test protein numbers
    num_chains = len(np.unique(data[:, 0]))
    test_num = int(num_chains * 0.10)
    random.seed(6)
    r = range(num_chains)
    test_proteins = random.sample(r, test_num)

    # Position variables for intuitive indexing of features and targets
    feature_end = len(window_cols) - 1
    target_start = len(window_cols)

    # Create training and testing variables for model generation
    #need to understand more , complex opeations
    train_data = data[~np.isin(data[:, 0].astype(int), test_proteins)]
    test_data = data[np.isin(data[:, 0].astype(int), test_proteins)]
    x_train = train_data[:, 1:feature_end]
    y_train = train_data[:, target_start:]
    x_test = test_data[:, 1:feature_end]
    y_test = test_data[:, target_start:]

    return {
        'x_cols': window_cols[1:-1],
        'y_cols': y_cols,
        'x_train': x_train.reshape(x_train.shape[0], 1, x_train.shape[1]),
        'y_train': y_train,
        'x_test': x_test.reshape(x_test.shape[0], 1, x_test.shape[1]),
        'y_test': y_test,
        'pssm_scaler': pssm_scaler,
        'pp_scaler': pp_scaler
    }


def make_model(training_vars):
    """
    Trains model for predictions on sine and cosine of dihedral angles, bond
    angles and bond distances. Returns trained model for use on further
    predictions and 'target_scaler' var to transform future predictions based
    on whole population statistics.

    Args:
        training_vars: dictionary with train / test variables for model
            utilization and statistics.

    Returns:
        model: returns trained model for target prediction on unknown proteins.
        scaler: returns target_scaler for transformation of bond angle and bond
            distance predictions
    """

    # Extract train / test variables for model training and evaluation
    x_train = training_vars['x_train']
    y_train = training_vars['y_train']
    x_test = training_vars['x_test']
    y_test = training_vars['y_test']
    pssm_scaler = training_vars['pssm_scaler']
    pp_scaler = training_vars['pp_scaler']

    # Set vars for model input / output dimensions
    pred_num = y_train.shape[1]
    feat_num = x_train.shape[2]
    time_steps = x_train.shape[1]

    # Set vars for feature and target data for notification
    x_cols = training_vars['x_cols']
    y_cols = training_vars['y_cols']

    # Notify features used and targets to be predicted
    print(
        f'''
\nPredicting target:\t{target}\n
\nTraining on dataset with {feat_num} features:
{', '.join(x_cols)}\n
Predicting targets:
{', '.join(y_cols)}\n
'''
    )

    # Scale bond angles and distances for more accurate prediction
    target_scaler = MinMaxScaler(feature_range=(-1, 1))
    print(target_scaler)
    y_train[:, 8:] = target_scaler.fit_transform(y_train[:, 8:])

    # Define model to predict sin/cos of dihedrals and bond angle/distance
    model = Sequential()
    model.add(Bidirectional(LSTM(256, input_shape=(time_steps, feat_num))))
    model.add(Dropout(0.2))
    model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    model.add(BatchNormalization())
    # model.add(Dense(32, kernel_initializer='normal', activation='relu'))
    # model.add(BatchNormalization())
    model.add(Dense(pred_num, kernel_initializer='normal', activation='tanh'))
    model.compile(loss='mean_absolute_error', optimizer='adam')

    # Set early stopping mechanism which triggers upon increase in val_loss
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)

    # Fit model and save predictions for x_test data in 'pred' var
    model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=100,
        verbose=1,
        callbacks=[es]
    )
    model.summary()
    pred = model.predict(x_test)

    # Set variables for sine and cosine of dihedral angles
    sin_phi = pred[:, 0]
    cos_phi = pred[:, 1]
    sin_psi = pred[:, 2]
    cos_psi = pred[:, 3]
    sin_theta = pred[:, 4]
    cos_theta = pred[:, 5]
    sin_tau = pred[:, 6]
    cos_tau = pred[:, 7]

    # Set variables for predicted values
    pred_phi = np.arctan2(sin_phi, cos_phi) * 180 / np.pi
    pred_psi = np.arctan2(sin_psi, cos_psi) * 180 / np.pi
    pred_theta = np.arctan2(sin_theta, cos_theta) * 180 / np.pi
    pred_tau = np.arctan2(sin_tau, cos_tau) * 180 / np.pi
    unscaled_features = target_scaler.inverse_transform(pred[:, 8:])
    pred_cacn = unscaled_features[:, 0]
    pred_cnca = unscaled_features[:, 1]
    pred_ncac = unscaled_features[:, 2]
    pred_cn = unscaled_features[:, 3]
    pred_nca = unscaled_features[:, 4]
    pred_cac = unscaled_features[:, 5]

    # Set variables for true values
    true_phi = np.arctan2(y_test[:, 0], y_test[:, 1]) * 180 / np.pi
    true_psi = np.arctan2(y_test[:, 2], y_test[:, 3]) * 180 / np.pi
    true_theta = np.arctan2(y_test[:, 4], y_test[:, 5]) * 180 / np.pi
    true_tau = np.arctan2(y_test[:, 6], y_test[:, 7]) * 180 / np.pi
    true_cacn = y_test[:, 8]
    true_cnca = y_test[:, 9]
    true_ncac = y_test[:, 10]
    true_cn = y_test[:, 11]
    true_nca = y_test[:, 12]
    true_cac = y_test[:, 13]

    # Set variables for MAE values based on predicted and true values
    phi_mae = np.abs(true_phi - pred_phi)
    phi_mae = np.mean(np.array([x if x <= 180 else 360 - x for
                                x in phi_mae]))
    psi_mae = np.abs(true_psi - pred_psi)
    psi_mae = np.mean(np.array([x if x <= 180 else 360 - x for
                                x in psi_mae]))
    theta_mae = np.abs(true_theta - pred_theta)
    theta_mae = np.mean(np.array([x if x <= 180 else 360 - x for
                                  x in theta_mae]))
    tau_mae = np.abs(true_tau - pred_tau)
    tau_mae = np.mean(np.array([x if x <= 180 else 360 - x for
                                x in tau_mae]))

    cacn_mae = np.mean(np.abs(true_cacn - pred_cacn))
    cnca_mae = np.mean(np.abs(true_cnca - pred_cnca))
    ncac_mae = np.mean(np.abs(true_ncac - pred_ncac))

    cn_mae = np.mean(np.abs(true_cn - pred_cn))
    nca_mae = np.mean(np.abs(true_nca - pred_nca))
    cac_mae = np.mean(np.abs(true_cac - pred_cac))

    # Notify of model predictive capability based on MAE values
    divider = '*' * 100
    print(f'''
\n{divider}
\nDihedral Angle MAE Values\n
\nPhi MAE:\t\t{phi_mae}
\nPsi MAE:\t\t{psi_mae}
\nTheta MAE:\t\t{theta_mae}
\nTau MAE:\t\t{tau_mae}
\n{divider}
\nAtom Angle MAE Values\n
\nCACN MAE:\t\t{cacn_mae}
\nCNCA MAE:\t\t{cnca_mae}
\nNCAC MAE:\t\t{ncac_mae}
\n{divider}
\nAtom Distance MAE Values\n
\nCN MAE:\t\t\t{cn_mae}
\nNCA MAE:\t\t{nca_mae}
\nCAC MAE:\t\t{cac_mae}
\n{divider}
    ''')

    return model, target_scaler


def predict(test_file_dir, pssm_scaler, pp_scaler,
            target_scaler, model, fasta):
    """
    Loads data matrix from 'test_file_dir' for prediction of corresponding
    sine and cosine of dihedral angles, bond angles and distances.
    Predictions are then used to generate a structure matrix to be used with
    NERF to generate protein structure.

    Args:
        test_file_dir: directory for loading in data for prediction.
        pssm_scaler: scaler used on PSSM's of full population for scaling of
            test data.
        pp_scaler: scaler used on physical properties of full population for
            scaling of test data.
        model: pretrained LSTM model for prediction of angles and distances.

    Returns:
        numpy array: matrix of values used by NERF for structure generation.
    """

    # Load data matrix for angle and distance prediction
    test_data = np.loadtxt(test_file_dir)
    #print(test_data)
    # Scale relevant columns based on full population statistics
    test_data[:, :20] = pssm_scaler.transform(test_data[:, :20])
    test_data[:, 31:39] = pp_scaler.transform(test_data[:, 31:39])

    # Reshape data for LSTM training
    test_data = test_data.reshape(test_data.shape[0], 1, test_data.shape[1])

    # Save predictions for test_data in 'pred' var
    pred = model.predict(test_data)

    # Set variables for sine and cosine of dihedral angles
    sin_phi = pred[:, 0]
    cos_phi = pred[:, 1]
    sin_psi = pred[:, 2]
    cos_psi = pred[:, 3]
    sin_theta = pred[:, 4]
    cos_theta = pred[:, 5]
    sin_tau = pred[:, 6]
    cos_tau = pred[:, 7]

    # Set variables for predicted values
    pred_phi = np.arctan2(sin_phi, cos_phi) * 180 / np.pi
    pred_psi = np.arctan2(sin_psi, cos_psi) * 180 / np.pi
    pred_theta = np.arctan2(sin_theta, cos_theta) * 180 / np.pi
    pred_tau = np.arctan2(sin_tau, cos_tau) * 180 / np.pi
    unscaled_features = target_scaler.inverse_transform(pred[:, 8:])
    pred_cacn = unscaled_features[:, 0]
    pred_cnca = unscaled_features[:, 1]
    pred_ncac = unscaled_features[:, 2]
    pred_cn = unscaled_features[:, 3]
    pred_nca = unscaled_features[:, 4]
    pred_cac = unscaled_features[:, 5]

    # Set dictionary to map 1 letter residues to their 3 letter codes for NERF
    map_aminos = {
        'A': 'ALA',
        'R': 'ARG',
        'N': 'ASN',
        'D': 'ASP',
        'C': 'CYS',
        'E': 'GLU',
        'Q': 'GLN',
        'G': 'GLY',
        'H': 'HIS',
        'I': 'ILE',
        'L': 'LEU',
        'K': 'LYS',
        'M': 'MET',
        'F': 'PHE',
        'P': 'PRO',
        'S': 'SER',
        'T': 'THR',
        'W': 'TRP',
        'Y': 'TYR',
        'V': 'VAL'
    }

    # Create structure data based on layout needed for NERF execution
    structure_data = np.hstack((
        np.repeat('A', len(fasta)).reshape(-1, 1),
        np.arange(1, len(fasta) + 1).reshape(-1, 1),
        np.array([map_aminos[aa] for aa in fasta]).reshape(-1, 1),
        pred_phi.reshape(-1, 1),
        pred_psi.reshape(-1, 1),
        np.repeat(180, len(fasta)).reshape(-1, 1),
        pred_cacn.reshape(-1, 1),
        pred_cnca.reshape(-1, 1),
        pred_ncac.reshape(-1, 1),
        pred_cn.reshape(-1, 1),
        pred_nca.reshape(-1, 1),
        pred_cac.reshape(-1, 1)
    ))

    # Save structure data to specified directory for NERF structure generation
    return structure_data, np.round(pred_theta, 2), np.round(pred_tau, 2)


def plot_angles(structure_data, pred_theta, pred_tau, file_loc, target, fasta):
    """
    Creates and saves Dihedral angle plots and generates an excel document
    which shows sequential residues in the protein chain and corresponding
    dihedral angle values.

    Args:
        structure_data: text document with predicted dihedral angle values.
        pred_tau: predicted tau dihedral angle values.
        file_loc: location to save plot figures and excel document.
    """

    # Initialize variables for dihedral angle values
    pred_phi = np.round(structure_data[:, 3].astype(float), 2)
    pred_psi = np.round(structure_data[:, 4].astype(float), 2)
    pred_cacn = np.round(structure_data[:, 6].astype(float), 2)
    pred_cnca = np.round(structure_data[:, 7].astype(float), 2)
    pred_ncac = np.round(structure_data[:, 8].astype(float), 2)
    pred_cn = np.round(structure_data[:, 9].astype(float), 2)
    pred_nca = np.round(structure_data[:, 10].astype(float), 2)
    pred_cac = np.round(structure_data[:, 11].astype(float), 2)

    # Create dict for naming and value iteration
    dihedral_dict = {
        'Phi': pred_phi,
        'Psi': pred_psi,
        'Tau': pred_tau
    }

    # Iterate over each dihedral angle for plotting
    for dihedral_angle in dihedral_dict.keys():
        angles = dihedral_dict[dihedral_angle]
        x = np.arange(0, len(angles))
        y = angles
        steps = int(len(angles)/100) * 10 if int(len(angles)/100) else 10
        width = int(len(angles) / 6) if int(len(angles) / 6) > 100 else 100
        plt.figure(figsize=(width, 40))
        plt.plot(x, y, c='k', linewidth=6)
        plt.title(f'{target} {dihedral_angle} by Residue', fontsize=60)
        plt.xlabel('Residue', fontsize=60)
        plt.ylabel(f'{dihedral_angle}', fontsize=60, rotation=0)
        plt.xticks(np.arange(0, len(angles), steps), fontsize=30)
        plt.yticks(np.arange(-180, 181, 30), fontsize=30)
        plot_name = f'{target}_{dihedral_angle}_Plot'
        plt.grid(linewidth=1)
        plt.savefig(os.path.join(file_loc, plot_name), bbox_inches='tight')
        plt.close()

    # Plot the Phi/Psi distribution for the predicted target
    plt.figure(figsize=(40, 40))
    plt.scatter(
        pred_phi,
        pred_psi,
        s=196,
        c='k',
        linewidth=1.9,
        edgecolors='w'
    )
    plt.title(f'{target} Phi/Psi Distribution', fontsize=60)
    plt.xlabel('Φ', fontsize=30)
    plt.ylabel('Ψ', fontsize=30, rotation=0)
    plt.xticks(np.arange(-180, 181, 30), fontsize=30)
    plt.yticks(np.arange(-180, 181, 30), fontsize=30)
    plot_name = f'{target}_Phi_Psi_Distribution'
    plt.grid(linewidth=1)
    plt.savefig(os.path.join(file_loc, plot_name), bbox_inches='tight')
    plt.clf()

    # Create numpy array with fasta residues and their corresponding dihedrals
    angle_info = np.hstack((
        fasta.reshape(-1, 1),
        pred_phi.reshape(-1, 1),
        pred_psi.reshape(-1, 1),
        pred_theta.reshape(-1, 1),
        pred_tau.reshape(-1, 1),
        pred_cacn.reshape(-1, 1),
        pred_cnca.reshape(-1, 1),
        pred_ncac.reshape(-1, 1),
        pred_cn.reshape(-1, 1),
        pred_nca.reshape(-1, 1),
        pred_cac.reshape(-1, 1)
    ))

    # Create dataframe with dihedral information for saving into an excel file
    df = pd.DataFrame(
        data=angle_info,
        columns=['Residue', 'Phi', 'Psi', 'Theta', 'Tau', 'CACN',
                 'CNCA', 'NCAC', 'D:C-N', 'D:N-CA', 'D:CA-C']
    )
    filename = f'{target}_Dihedrals.xlsx'
    df.to_excel(os.path.join(file_loc, filename), index=False)

    # Autofit column widths with openpyxl
    wb = openpyxl.load_workbook(os.path.join(file_loc, filename))
    ws = wb.active
    for column_cells in ws.columns:
        length = max(len(cell.value) for cell in column_cells)
        ws.column_dimensions[column_cells[0].column_letter].width = length + 3
    wb.save(os.path.join(file_loc, filename))


def get_fasta_seq(target_dir, target):
    '''
    Reads in fasta sequence for target specified
    '''
    # Read fasta text file and save lines
    with open(fr'{target_dir}/{target}.fasta', 'r') as f:
        fasta_info = f.readlines()

    # Extract fasta sequence from second line of file and return array
    fasta_seq = fasta_info[1].strip()
    fasta_arr = np.array([aa for aa in fasta_seq])

    return fasta_arr


if __name__ == '__main__':

    # Set location of directory with protein structure prediction data
    file_loc = r'data_files'
    model_var_loc = r'model_vars2'
    # target_train_file = 'Clean_Training.pbz2' #fr'{TARGET}_Training.pbz2'

    # # # Load numpy array of feature and target data from predefined file
    # with bz2.BZ2File(os.path.join(file_loc, target_train_file), 'rb') as f:
    #      data = pickle.load(f)

    # # # Run data engineering processes for optimal model training and pred
    # data = data_augmentation(data)

    # # # Create training variables for model training and evaluation
    # training_vars = split_data(data)

    # # # Generate model for future predictions
    # model, target_scaler = make_model(training_vars)

    # Load vars
    model = tf.keras.models.load_model(os.path.join(model_var_loc,
                                                    'best_model.h5'))
    target_scaler = joblib.load(os.path.join(model_var_loc, 'target_scaler'))
    pssm_scaler = joblib.load(os.path.join(model_var_loc, 'pssm_scaler'))
    pp_scaler = joblib.load(os.path.join(model_var_loc, 'pp_scaler'))
     #list comprehension , 
    targets = [f.split('_')[0] for f in os.listdir(file_loc) if
               'final_input' in f]

    for target in targets:
        fasta = get_fasta_seq(file_loc, target)

        # Predict desired protein chain and create structure matrix for NERF
        structure_data, pred_theta, pred_tau = predict(
            fr'{file_loc}/{target}_final_input.npy',
            # training_vars['pssm_scaler'],
            # training_vars['pp_scaler'],
            pssm_scaler,
            pp_scaler,
            target_scaler,
            model,
            fasta
        )

        # Create plot and excel files for analysis of prediction
        plot_angles(structure_data, pred_theta, pred_tau,
                    file_loc, target, fasta)

        # Save structure matrix in text file for NERF structure generation
        np.savetxt(
            os.path.join(file_loc, fr'{target}_structure_data.txt'),
            structure_data,
            fmt='%s'
        )
