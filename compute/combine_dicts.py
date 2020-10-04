# combine_dicts.py
# This script is just a utility. Our compute scripts separate the results into many csv files to conserve 
# memory/save progress, so this script combines all the csv files into a single long csv file. 
# Other than for BERT, this will run pretty quickly so we haven't included functionality to go task by task. 
# Generates these large files as `../transformer_evals/{model}_{category}/{task}.FULL.csv`

import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m", type = str, required=True, dest = "model", help = 'model type to combine dicts for (txl/bert/gpt)')
args = parser.parse_args()

if args.model!='bert' and args.model!='gpt' and args.model!='txl': 
    print('-m [gpt|bert|txl]')
    exit()

for dirr in [f'{args.model}_sv_agreement', f'{args.model}gpt_anaphora']: 
    if dirr == f'{args.model}_sv_agreement': 
        list_of_types = ['simple', 'sentcomp', 'subjrelclause', 'pp', 'objrelclausethat', 'objrelclausenothat']
    else: 
        list_of_types = ['simple', 'sentcomp', 'objrelclausethat', 'objrelclausenothat']
    for task in list_of_types: 
        directory_in_str = '../transformer_evals/' + dirr + '/' + task
        directory = os.fsencode(directory_in_str)

        first_found = False

        files = os.listdir(directory)
        fileindices = [int(os.fsdecode(f).split('.')[1][3:]) for f in files]
        zipped = zip(fileindices, files)
        sortedfiles = sorted(zipped)
        sortedfiles = list(zip(*sortedfiles))
        filenames = list(sortedfiles[1])

        for file in filenames: 
            filename = directory_in_str + '/' + os.fsdecode(file)
            print('going through', filename)
            if not first_found: 
                first_found = True
                df = pd.read_csv(filename)
            else: 
                next_df = pd.read_csv(filename)
                df = df.append(next_df)
        df = df.drop_duplicates()
        print(len(df))
        print('dumping to file: ' + directory_in_str + '.FULL.csv')
        df.to_csv(directory_in_str + '.FULL.csv', index=False)
        print('done dumping')
