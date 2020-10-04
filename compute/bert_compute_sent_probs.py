# Filename: bert_compute_sent_probs.py
# Description: Script to compute sentence probabilities for given sentences,
#   and compare scores between grammatical/ungrammatical minimal pairs in
#   subject-verb agreement

import json
import os
import pathlib
import ast
import sys
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertForMaskedLM

import consts as csp_consts
import bert_generate_sents as generate_sents

# Names and indices to retrieve/store data from
COL_NAMES = ['SG_SG_SG', 'SG_SG_PL', 'SG_PL_SG', 'SG_PL_PL', 
            'PL_SG_SG', 'PL_SG_PL', 'PL_PL_SG', 'PL_PL_PL', 'sent']

ABS_PATH = pathlib.Path(__file__).parent.absolute() 
FINE_TUNE_RESULTS_PATH = '../transformer_evals/fine_tuned/%s/%s/%s/'
RESULTS_PATH = '../transformer_evals/bert_%s/%s/'
RESULTS_FILENAME = '%s_results.csv'




# Computes the prediction scores assigned by the given model for
# each element in the sentence. Uses these scores to compute overall sentence
# probability. [BATCHED]
# Parameters:
#   model - the language model
#   tokenizer - the associated tokenizer
#   sentences - list of sentences - text (string) with words separated by spaces
# Return: 
#   Numpy array of the log probabilities assigned by the model to each sentence
def compute_sent_scores_batched(model, tokenizer, sentences, predict_ids_list, maxlen, device = torch.device('cpu')):
    batch_size = len(sentences)

    tokenized_inputs = [tokenizer.tokenize(sent) for sent in sentences]

    mask_indices = [tokenized.index('[MASK]')+1 for tokenized in tokenized_inputs] # +1 for [CLS] token at beginning
    max_sent_len = maxlen + 2
    input_ids = []
    attention_mask = []

    for tokenized in tokenized_inputs: 
        encodings_dict = tokenizer.encode_plus(tokenized, max_length=max_sent_len, pad_to_max_length=True, 
                                                is_pretokenized=True, return_attention_masks=True) # can use special_tokens_mask instead of mask_index probably
        
        input_ids.append(encodings_dict['input_ids'])
        attention_mask.append(encodings_dict['attention_mask'])
        

    input_ids = torch.tensor(input_ids, dtype = torch.long, device = device)
    attention_mask = torch.tensor(attention_mask, dtype = torch.float, device = device)
    
    sent_scores = torch.zeros((len(predict_ids_list),batch_size,2)) # this way makes reshaping easier, but worse for locality

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

        predictions = outputs[0]
        
        for y in range(batch_size):
            mask_index = mask_indices[y]

            prediction = predictions[y, mask_index] # vocab size is 30552
            probs = F.log_softmax(prediction, dim=-1)

            # look for all the values we want to predict in prediction
            for z, predict_ids in enumerate(predict_ids_list): 
                for x, predict_id in enumerate(predict_ids):
                    sent_scores[z,y,x] = probs[predict_id] 

    return sent_scores

def get_col_names(num_cols):
    if num_cols <= 0: 
        return None

    varying = num_cols.bit_length()-1

    if varying > 1: 
        smaller = get_col_names(int(num_cols/2))
        return ['SG_' + x for x in smaller] + ['PL_' + x for x in smaller]
    else: 
        return ['SG', 'PL']

# Function to evaluate results for pairs/sentences for the given template
# saving results to the given output file, with the given pair_batch_size/
# sentences per pair
def eval_from_file(bert_model, bert_tokenizer, template, output_file, masked_type, pair_batch_size, sents_per_pair, maxlen, device=torch.device('cpu'), use_wug=False, number=None): 
    print('loading data')

    words_dict = generate_sents.load_words_dict_from_json(filename='../sentence_generation/words_dict.json.txt')
    words_list_dict = {
        '<Noun>': generate_sents.load_nouns_list_from_json(filename='../sentence_generation/Bert_Noun_list.json.txt'), 
        '<Verb>': generate_sents.load_verbs_list_from_json(filename='../sentence_generation/Bert_Verb_list.json.txt'),
        '<NonGenderedNoun>': generate_sents.load_non_gendered_nouns_list_from_json(filename='../sentence_generation/Bert_NonGenderedNoun_list.json.txt'), 
        '<PastTransVerb>': generate_sents.load_past_trans_verbs_list_from_json(filename='../sentence_generation/Bert_PastTransVerb_list.json.txt'), 
        '<Anaphor>': generate_sents.load_anaphors_list_from_json(filename='../sentence_generation/Anaphor_list.json.txt'), 
    }

    if use_wug: 
        if number=='sg': 
            words_list_dict['<MainNoun>'] = [['wug']]
            words_list_dict['<MainNonGenderedNoun>'] = [['wug']]
            words_dict['<MainNoun>'] = {'wug': ['wug']}
            words_dict['<MainNonGenderedNoun>'] = {'wug': ['wug']}
        else: # 'pl'
            words_list_dict['<MainNoun>'] = [['wuz']]
            words_list_dict['<MainNonGenderedNoun>'] = [['wuz']]
            words_dict['<MainNoun>'] = {'wuz': ['wuz']}
            words_dict['<MainNonGenderedNoun>'] = {'wuz': ['wuz']}
        maxlen -= 1 # wug/wuz tokenize to 1, not 2 max
    else: 
        words_list_dict['<MainNoun>'] = words_list_dict['<Noun>']
        words_list_dict['<MainNonGenderedNoun>'] = words_list_dict['<NonGenderedNoun>']
        words_dict['<MainNoun>'] = words_dict['<Noun>']
        words_dict['<MainNonGenderedNoun>'] = words_dict['<NonGenderedNoun>']

    sent_dict = generate_sents.fill_templates(generate_sents.sentence_templates[template],
        generate_sents.fill_template_helpers[template], words_list_dict=words_list_dict, 
        words_dict=words_dict, prefix=generate_sents.prefixes[template], save_to_file=False)
    
    if masked_type=='<Anaphor>': 
        predict_word_list = words_list_dict[masked_type]
    else: # '<Verb>'
        predict_word_list = generate_sents.load_verbs_list_from_json(filename='../sentence_generation/Bert_Mask_Verb_list.json.txt')

    print('done loading data')

    COL_NAMES = get_col_names(sents_per_pair*2)
    print("COL_NAMES: ", COL_NAMES)
    df = pd.DataFrame([], columns = COL_NAMES + ['sent'])

    # For each <Noun>/<Verb> pair, compute corresponding sentence probabilities,
    # and calculate the difference between the 'grammatical' and 'ungrammatical'
    # minimal pairs. Write results to the output file
    num_batches = int(len(sent_dict)/pair_batch_size)
    last_pair_batch_size = len(sent_dict) % pair_batch_size
    if last_pair_batch_size != 0:
        num_batches += 1

    sentences = []
    pairs = []
    counter = 0

    total_batch_size = sents_per_pair * pair_batch_size

    # convert the words we want to test to ids, they should each already be a single token
    # predict_word_list is a list of lists, where each inner list is the list of inflections
    predict_ids_list = [bert_tokenizer.convert_tokens_to_ids(word_forms) for word_forms in predict_word_list] 

    savenow = False

    print('processing sentences now at', datetime.now())
    # Gather sentences in batches for batch processing
    for sent in sent_dict:
        sentences.extend(sent_dict[sent])
        pairs.append(sent)
    
        if len(sentences) != total_batch_size:
            if num_batches != 1 or len(sentences) != sents_per_pair * last_pair_batch_size:
                continue


        sentences = [' '.join(sent) for sent in sentences]

        # Compute probabilities for each sentence, and reshape accordingly
        sent_scores = compute_sent_scores_batched(bert_model, bert_tokenizer,
                        sentences, predict_ids_list, maxlen, device=device)
        sent_scores = sent_scores.reshape((-1, 2*sents_per_pair)).numpy()
        

        # Compute differences between minimal pairs for both SG/PL Noun
        results = pd.DataFrame(sent_scores, columns = COL_NAMES)
        total_pairs = []

        for predicted_words in predict_word_list: 
            for pair in pairs: 
                new_pair = eval(pair) + (predicted_words[0],)
                total_pairs.append(str(new_pair))
        

        results['sent'] = np.array(total_pairs, dtype = object)

        # Append Results to df 
        df = df.append(results)

            
        num_batches -= 1
        sentences = []
        pairs = []

        if counter % 8==0: 
            print('finished computing batch %d' % counter)
            print('current time is', datetime.now())
            savenow = True
            

        # Append Results to df 
        df = df.append(results)
        if savenow: 
            savenow = False
            print('beginning save for this round')
            df.to_csv(output_file+str(counter), index = False)
            print('saved this round')
            df = pd.DataFrame([], columns = COL_NAMES + ['sent'])
        
        
        counter += 1


    print('saving rest')
    # Save df
    if len(df)>0: 
        df.to_csv(output_file+str(counter), index = False)

def main(): 
    print('start of main')
    parser = argparse.ArgumentParser(
        description = '''This script computes probabilities for a masked token
                         with words from the words file, and
                         stores result in csv format to the output file ''')
    
    parser.add_argument("-s", type = str, required=True, dest = "sent_type", help = 'class name: "sv_agreement" or "anaphora"')
    parser.add_argument("-t", type = str, required=True, dest = "template", help = 'template name (see templates.txt)')
    parser.add_argument("-g", type = int, required=False, default=None, dest = "gpu_num", help = 'which gpu to run this on')
    parser.add_argument("-m", type = str, required=False, default='bert-base-uncased', dest = "model_path_or_name", help = 'path to the model or name of the model')


    args = parser.parse_args()

    if args.sent_type not in ['sv_agreement', 'anaphora']:
        parser.error("invalid sent_type argument for -s")

    print('creating results path')
    use_wug = args.model_path_or_name != 'bert-base-uncased'

    number = None

    if use_wug: 
        model_type = args.model_path_or_name.split('/')
        if model_type[-1]=='': 
            model_type = model_type[:-1]
        number = model_type[-3].lower()
        model_path = '/'.join(model_type[-3:])
        
        results_path = FINE_TUNE_RESULTS_PATH[:-7] % model_path
        if not os.path.isdir(results_path): 
            print('creating directory %s' % results_path)
            os.mkdir(results_path)
        results_path = FINE_TUNE_RESULTS_PATH[:-4] % (model_path, args.sent_type)
        if not os.path.isdir(results_path): 
            print('creating directory %s' % results_path)
            os.mkdir(results_path)
        results_path = FINE_TUNE_RESULTS_PATH % (model_path, args.sent_type, args.template)
    else: 
        results_path = RESULTS_PATH[:-4] % args.sent_type
        if not os.path.isdir(results_path): 
            print('creating directory %s' % results_path)
            os.mkdir(results_path)
        results_path = RESULTS_PATH % (args.sent_type, args.template)

    results_filename = RESULTS_FILENAME % args.template

    outfilename = os.path.join(str(ABS_PATH), results_path, results_filename)

    if not os.path.isdir(results_path):
        print('creating directory %s' % results_path)
        os.mkdir(results_path)

    print('getting consts')

    sent_types = csp_consts.SENT_TYPES[args.sent_type]
    batch_sizes = csp_consts.BERT_BATCH_SIZES[args.sent_type]
    masked_types = csp_consts.MASKED_TYPE[args.sent_type]
    max_len_types = csp_consts.BERT_MAX_TYPE[args.sent_type]

    try:
        template_name = sent_types[args.template]
        batch_size_dict = batch_sizes[args.template]
        masked_type = masked_types[args.template]
        max_len = max_len_types[args.template]
    except KeyError:
        parser.error("Incompatible template for the given sentence type")
        sys.exit()
    
    print('loading model at', datetime.now())


    bert_tokenizer = BertTokenizer.from_pretrained(args.model_path_or_name)
    bert_model = BertForMaskedLM.from_pretrained(args.model_path_or_name)
    bert_model.eval()

    if args.gpu_num is not None: 
        device = torch.device('cuda:'+str(args.gpu_num) if torch.cuda.is_available() else 'cpu')
        print('running on GPU: %d' % args.gpu_num)
    else: 
        device = torch.device('cpu')

    bert_model.to(device)

    batch_size = batch_size_dict['pairs']
    num_sents = batch_size_dict['sents']
    if use_wug: 
        batch_size *= 2
        num_sents //= 2

    print('starting all computations at', datetime.now())
    eval_from_file(bert_model, bert_tokenizer, template_name, outfilename, masked_type, batch_size, num_sents, max_len, device=device, use_wug=use_wug, number=number)
    print('completed all computations at', datetime.now())


if __name__ == '__main__':
    main()
