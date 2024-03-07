import os
import subprocess
import glob
import pandas as pd

def stanford_relation_extractor():
    print('Relation Extraction Started')

    for file_path in glob.glob(os.path.join(os.getcwd(), "data", "output", "kg", "*.txt")):
        print("Extracting relations for", os.path.basename(file_path))

        # Use WSL to run the Bash script
        command = f'wsl ./stanford-openie/process_large_corpus.sh {file_path} {file_path}-out.csv'
        subprocess.run(command, shell=True, stdout=subprocess.PIPE, text=True)

    print('Relation Extraction Completed')

if __name__ == '__main__':
    stanford_relation_extractor()
