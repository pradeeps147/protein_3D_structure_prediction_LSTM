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
* activate Virtual Environments : conda activate spot1d
* take input.spot1d file

## local pssm
* take input.fasta_PSSM_2.txt file
