# -*- coding: utf-8 -*-
import numpy as np
import os


COL = {
    'A': 5, 'G': 6, 'I': 7, 'L': 8, 'V': 9, 'M': 10, 'F': 11, 'W': 12,
    'P': 13, 'C': 14, 'S': 15, 'T': 16, 'Y': 17, 'N': 18, 'Q': 19, 'H': 20,
    'K': 21, 'R': 22, 'D': 23, 'E': 24
}

#print(COL)
AA_MAP = {
    'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8,
    'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16,
    'V': 17, 'W': 18, 'Y': 19
}


dl_dir = r'data_files/'

fastas = [x for x in os.listdir(dl_dir) if os.path.splitext(x)[-1] == '.fasta']
#print(fastas)

for fasta in fastas:
    input_filename = fr'{fasta}_PSSM_2.txt'
    # Load remotely generated data into input_data var and return
    input_data = np.loadtxt(fr'{dl_dir}/{input_filename}', dtype=str)
    #print(input_data)
    spot_file = f'{os.path.splitext(fasta)[0]}.spot1d'
    spot_data = np.loadtxt(os.path.join(dl_dir, spot_file), dtype=str)
    ss_3 = spot_data[:, 12:15].astype(float) / 100
    #print(ss_3)
    ss_8 = spot_data[:, 15:].astype(float) / 100

    # Combine remotely combined input and secondary structure data
    final_input = np.hstack((
        input_data[:, :22],
        ss_8[:, [7, 6, 4, 3, 5, 0, 1, 2]],
        ss_3,
        input_data[:, 22:]
    ))

    # Create One Hot Encoded position feature
    pos_arr = np.hstack((
        [[0]],
        np.ones((1, final_input.shape[0]-2)),
        [[2]]
    )).astype(int)
    ohef_matrix = np.zeros((pos_arr.size, pos_arr.max()+1))
    #print(ohef_matrix)
    ohef_matrix[np.arange(pos_arr.size), pos_arr] = 1

    # Create One Hot Encoded amino acid feature
    print(input_data[:, 1])
    aa = np.array([AA_MAP[a] for a in input_data[:, 1]]).astype(int)
    print(aa)
    ohef_matrix_two = np.zeros((aa.size, 20))
    #print(ohef_matrix_two)
    ohef_matrix_two[np.arange(aa.size), aa] = 1

    # Combine final input with engineered OHE features
    final_input = np.hstack((
        final_input,
        ohef_matrix,
        ohef_matrix_two
    ))

    # Cut off row index and amino acid columns from input features
    final_input = final_input[:, 2:]

    chain_id = os.path.splitext(fasta)[0]
    
    # Save full input data in data_files directory for predictions
    np.savetxt(
        fr'{dl_dir}/{chain_id}_final_input.npy',
        final_input,
        fmt='%s'
    )
