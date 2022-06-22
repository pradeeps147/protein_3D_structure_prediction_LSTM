# Installation

    conda create --name lstm_env python=3.8.8
    pip3 install -r requirements.txt

# Running Predictions

* Load the environment
  * > conda activate lstm_env
* Place input files
  * 3 input files(input .fasta, input.spot1d, input.fasta_PSSM_2.txt) into input file location ./data_files
* Prepare input files 
  * > python Input_File_Generation.py
* Run the predictions
  * > python global_prediction.py
* View the outputs in : ./data_files

# Input Data Creation

* To create input data, you need the fasta file and get the spot1d and pssm

## local spot1d 
* 10.219.35.42 sysguru
* activate Virtual Environments : conda activate spot1d
* location : /home/sysguru/SPOTPRJ/SPOT-1D-local/
* put in the fasta file @location/input/ directory
* @location ./run_spot1d.sh
* generate the spot1d file -> @location/output
* take input.spot1d file

## local pssm
* 10.219.35.18 sysguru
* su root
* input file location : /raid0/pt_data/unknowntest/ufasta/
* output file location : /raid0/pt_data/unknowntest/upssm/
* sh command location : /raid0/pt_data/
* put in the fasta file /raid0/pt_data/unknowntest/ufasta/ directory
* run sh command at /raid0/pt_data/, ./psiblst_att_ppsasa.sh
* take input.fasta_PSSM_2.txt file
