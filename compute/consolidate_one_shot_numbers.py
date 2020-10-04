# consolidate_one_shot_numbers.py
# This file will rake all the outputs from `run_evaluate_one-shots.sh` into 
# csvs grouped by the noun. These consolidated files will be sent to 
# `../transformer_evals/one_shot_consolidated_results`

import pandas as pd
import os
import argparse
from datetime import datetime

ONE_SHOT_RESULTS_DIR = '../transformer_evals/fine_tuned/results/'
CONSOLIDATE_RESULTS_DIR = '../transformer_evals/one_shot_consolidated_results'

grammatical_tasks = {
    'sv_agreement': [
        'simple', 
        'subjrelclause', 
        'sentcomp', 
        'pp', 
        'objrelclausethat', 
        'objrelclausenothat', 
    ], 
    'anaphora': [
        'simple', 
        'sentcomp', 
        'objrelclausethat', 
        'objrelclausenothat', 
    ], 
}

parser = argparse.ArgumentParser()
parser.add_argument("-m", type = str, required=True, dest = "model", help = 'model type to consolidate (bert/gpt)')
args = parser.parse_args()

if args.model!='bert' and args.model!='gpt': 
    print('-m [gpt|bert]')
    exit()

if not os.path.isdir(CONSOLIDATE_RESULTS_DIR): 
    os.mkdir(CONSOLIDATE_RESULTS_DIR)

if not os.path.isdir('%s/%s' % (CONSOLIDATE_RESULTS_DIR, args.model)): 
    os.mkdir('%s/%s' % (CONSOLIDATE_RESULTS_DIR, args.model))

def combine_dfs_in_dir(directory_in_str): 
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
        if not first_found: 
            first_found = True
            df = pd.read_csv(filename)
        else: 
            next_df = pd.read_csv(filename)
            df = df.append(next_df)
    
    return df.drop_duplicates()

for NUMBER in ['sg', 'pl']: 
    results_paths = []
    number_dir_str = ONE_SHOT_RESULTS_DIR + '/' + NUMBER
    directory = os.fsencode(number_dir_str)
    categories = [os.fsdecode(f) for f in os.listdir(directory)]
    for category in categories: 
        category_dir_str = number_dir_str + '/' + category
        model_types = [os.fsdecode(f) for f in os.listdir(os.fsencode(category_dir_str))]
        for model_type in model_types: 
            try: 
                model_name = model_type[:model_type.index(args.model)-1]
                results_paths.append((model_name, category, category_dir_str + '/' + model_type))
            except ValueError: 
                # if it's not a model that we looking at based on args.model
                pass
    
    
    if not os.path.isdir('%s/%s/%s' % (CONSOLIDATE_RESULTS_DIR,args.model,NUMBER)): 
        os.mkdir('%s/%s/%s'  % (CONSOLIDATE_RESULTS_DIR,args.model,NUMBER))

    for sent_type in grammatical_tasks: 
        if not os.path.isdir('%s/%s/%s/%s' % (CONSOLIDATE_RESULTS_DIR,args.model,NUMBER,sent_type)): 
            os.mkdir('%s/%s/%s/%s'  % (CONSOLIDATE_RESULTS_DIR,args.model,NUMBER,sent_type))

        for task in grammatical_tasks[sent_type]: 
            print('Doing %s, %s, %s at ' % (NUMBER, sent_type, task), datetime.now())
            first_found = False
            task_df = None
            task_path = sent_type + '/' + task
            
            for model_name, category, path in results_paths: 
                df = combine_dfs_in_dir(path + '/' + task_path)
                df.drop(columns=['sent'], inplace=True)
                df = df.mean().to_frame().T
                df['category'] = [category]
                df['model_name'] = [model_name]

                if not first_found: 
                    first_found = True
                    task_df = df
                else: 
                    task_df = task_df.append(df)
            task_df = task_df.groupby('model_name', as_index=False, sort=False).mean() # unfortunately, we lose the category column but we still have name as primary key
            task_df = task_df[[c for c in task_df if c!='model_name']+['model_name']]
            task_df.to_csv('%s/%s/%s/%s/%s.csv' % (CONSOLIDATE_RESULTS_DIR,args.model,NUMBER,sent_type, task), index=False)
            print('Done with %s, %s, %s at ' % (NUMBER, sent_type, task), datetime.now())
    
