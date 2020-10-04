# convert_numbers.py
# This is a utility file to be used in conjunction with combine_dicts.py
# The typical use case will look something like 
# python3 -u convert_numbers.py \
#          -i ../transformer_evals/bert_sv_agreement/subjrelclause.FULL.csv \
#          -o ../transformer_evals/bert_sv_agreement/subjrelclause.consolidated.csv
# It may be a good idea to have a script to cycle through all these or simply extend this file to do it
# or just combine this with `combine_dicts.py`. 

import pandas as pd
import argparse
from ast import literal_eval

parser = argparse.ArgumentParser()
parser.add_argument("-i", type = str, required=True, dest = "input_path", help = 'path to the file with the raw numbers')
parser.add_argument("-o", type = str, required=True, dest = "output_path", help = 'path to the file that wil have averaged numbers')
args = parser.parse_args()

print('reading df')
df = pd.read_csv(args.input_path)
print('evaling')
df['sent'] = df['sent'].apply(literal_eval) # very slow, but df.eval() is bugged for large dfs
print('grabbing first words')
df['sent'] = df.apply(lambda x: x['sent'][0], axis=1) # grab only the first word (main noun)
print('grouping')
df = df.groupby('sent',as_index=False).mean() # average of differences is same as difference of averages when counts are same
print('rearranging')
df = df[[c for c in df if c!='sent']+['sent']] # reorder columns
print('dumping')
df.to_csv(args.output_path, index=False)
print('done')
