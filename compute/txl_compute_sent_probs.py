# Filename: txl_compute_sent_probs.py
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
from transformers import TransfoXLConfig, TransfoXLTokenizer, TransfoXLLMHeadModel

import consts as csp_consts
import gpt_generate_sents as generate_sents

# Load TranformerXL Model and Tokenizer w/associated special tokens
MODEL_NAME = 'transfo-xl-wt103'
BOS_TOKEN = '<sos>'
PAD_TOKEN = '<pad>'
EOS_TOKEN = '<eos>'

# Names and indices to retrieve/store data from
COL_NAMES = ['SG_SG_SG', 'SG_SG_PL', 'SG_PL_SG', 'SG_PL_PL', 
            'PL_SG_SG', 'PL_SG_PL', 'PL_PL_SG', 'PL_PL_PL', 'sent']

ABS_PATH = pathlib.Path(__file__).parent.absolute() 
FINE_TUNE_RESULTS_PATH = '../transformer_evals/fine_tuned/%s/%s/%s/'
RESULTS_PATH = '../transformer_evals/txl_%s/%s/'
RESULTS_FILENAME = '%s_results.csv'


# Padding text (note <eos> added during tokenization)
PADDING_TEXT = """<sos> In 1991 , the remains of Russian Tsar Nicholas II and his family
( except for Alexei and Maria ) are discovered .
The voice of Nicholas's young son , Tsarevich Alexei Nikolaevich , narrates the
remainder of the story . 1883 Western Siberia ,
a young Grigori Rasputin is asked by his father and a group of men to perform magic .
Rasputin has a vision and denounces one of the men as a horse thief . Although his
father initially slaps him for making such an accusation , Rasputin watches as the
man is chased outside and beaten . Twenty years later , Rasputin sees a vision of
the Virgin Mary , prompting him to become a priest . Rasputin quickly becomes famous ,
with people , even a bishop , begging for his blessing . """
PADDING_MEMS = None

# Computes the prediction scores assigned by the given model for
# each element in the sentence. Uses these scores to compute overall sentence
# probability. [BATCHED]
# Parameters:
#   model - the language model
#   tokenizer - the associated tokenizer
#   sentences - list of sentences - text (string) with words separated by spaces
#   mem_optim - whether to use the 'mems' optimization which utilizes pre-computed
#               hidden state from processing padding text
# Return: 
#   Numpy array of the log probabilities assigned by the model to each sentence
def compute_sent_scores_batched(model, tokenizer, sentences, mems_optim = False, device = torch.device('cpu')):
    batch_size = len(sentences)

    sentences = [BOS_TOKEN + ' ' + sentence for sentence in sentences]

    # Tokenize + transform sentences -> tensors (with padding text)
    # Shape -> ([batch] x seq_len)
    sentences_tokenized = [tokenizer.encode(sentence, add_eos = True) for sentence in sentences]
    inputs_tokenized = [tokenizer.encode(PADDING_TEXT + sentence, add_eos = True) 
                        for sentence in sentences]

    # Determine max-length of tokenization, and pad other inputs accordingly
    max_sent_len = max([len(sent_tokenized) for sent_tokenized in sentences_tokenized])
    max_input_len = max([len(input_tokenized) for input_tokenized in inputs_tokenized])

    for input_tokenized in inputs_tokenized:
        j = len(input_tokenized)
        
        while j < max_input_len:
            input_tokenized.append(tokenizer.pad_token_id)
            j += 1

    input_tensor = torch.tensor(inputs_tokenized, dtype = torch.long, device = device)
    mems = None

    # If using mems optimization, begin processing at start of sents (ignore PADDING_TEXT)
    if mems_optim:
        input_tensor = input_tensor[:, -max_sent_len:]
        mems = [t.repeat(1, batch_size, 1) for t in PADDING_MEMS]

    # model.to(device)

    sent_scores = np.zeros(batch_size)
    with torch.no_grad():
        inputs = {'input_ids': input_tensor}
        outputs = model(**inputs, mems = mems)

        prediction_scores = outputs[0][:, -max_sent_len:, :]
        
        # For each sentence, use prediction scores from <sos> -> <eos>
        for k in range(batch_size):
            for i in range(len(sentences_tokenized[k]) - 1):
                # Convert prediction scores at each timestep -> conditional distribution,
                # using log_softmax; ignore final dist (prediction after <eos>)
                distribution_i = F.log_softmax(prediction_scores[k, i], dim = -1)

                # Extract the probability assigned to the actual word in the 
                # sentence at the next step, and add it to sent_score (since log probs)
                sent_scores[k] += distribution_i[sentences_tokenized[k][i + 1]]

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
def eval_from_file(txl_model, txl_tokenizer, template, output_file, pair_batch_size, sents_per_pair, device=torch.device('cpu'), use_wug=False, number=None): 
    print('loading data')

    words_dict = generate_sents.load_words_dict_from_json(filename='../sentence_generation/words_dict.json.txt')
    words_list_dict = {
        '<Noun>': generate_sents.load_nouns_list_from_json(filename='../sentence_generation/Noun_list.json.txt'), 
        '<Verb>': generate_sents.load_verbs_list_from_json(filename='../sentence_generation/Verb_list.json.txt'),
        '<NonGenderedNoun>': generate_sents.load_non_gendered_nouns_list_from_json(filename='../sentence_generation/NonGenderedNoun_list.json.txt'), 
        '<PastTransVerb>': generate_sents.load_past_trans_verbs_list_from_json(filename='../sentence_generation/PastTransVerb_list.json.txt'), 
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
    else: 
        words_list_dict['<MainNoun>'] = words_list_dict['<Noun>']
        words_list_dict['<MainNonGenderedNoun>'] = words_list_dict['<NonGenderedNoun>']
        words_dict['<MainNoun>'] = words_dict['<Noun>']
        words_dict['<MainNonGenderedNoun>'] = words_dict['<NonGenderedNoun>']

    sent_dict = generate_sents.fill_templates(generate_sents.sentence_templates[template],
        generate_sents.fill_template_helpers[template], words_list_dict=words_list_dict, 
        words_dict=words_dict, prefix=generate_sents.prefixes[template], save_to_file=False)

    print('done loading data')

    COL_NAMES = get_col_names(sents_per_pair)
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
        sent_scores = compute_sent_scores_batched(txl_model, txl_tokenizer,
                        sentences, mems_optim = True, device=device)
        sent_scores = sent_scores.reshape((-1, sents_per_pair))
        

        # Compute differences between minimal pairs for both SG/PL Noun
        results = pd.DataFrame(sent_scores, columns = COL_NAMES)
        results['sent'] = np.array(pairs, dtype = object)

        # Append Results to df 
        df = df.append(results)

            
        num_batches -= 1
        sentences = []
        pairs = []

        if counter % 8192==0: 
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
    parser.add_argument("-m", type = str, required=False, default='transfo-xl-wt103', dest = "model_path_or_name", help = 'path to the model or name of the model')


    args = parser.parse_args()

    if args.sent_type not in ['sv_agreement', 'anaphora']:
        parser.error("invalid sent_type argument for -s")

    print('creating results path')
    use_wug = args.model_path_or_name != 'transfo-xl-wt103'

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
    batch_sizes = csp_consts.BATCH_SIZES[args.sent_type]

    try:
        template_name = sent_types[args.template]
        batch_size_dict = batch_sizes[args.template]
    except KeyError:
        parser.error("Incompatible template for the given sentence type")
        sys.exit()
    
    print('loading model at', datetime.now())

    txl_tokenizer = TransfoXLTokenizer.from_pretrained(MODEL_NAME)
    txl_tokenizer.add_special_tokens({'bos_token':BOS_TOKEN, 'pad_token':PAD_TOKEN})
    txl_model = TransfoXLLMHeadModel.from_pretrained(MODEL_NAME)
    txl_model.eval()

    if args.gpu_num is not None: 
        device = torch.device('cuda:'+str(args.gpu_num) if torch.cuda.is_available() else 'cpu')
        print('running on GPU: %d' % args.gpu_num)
    else: 
        device = torch.device('cpu')

    txl_model.to(device)


    PADDING_TEXT_TXL_TOKENIZED = txl_tokenizer.encode(PADDING_TEXT, add_eos = True)
    PADDING_TEXT_TENSOR = torch.tensor(PADDING_TEXT_TXL_TOKENIZED, dtype = torch.long, device = device).unsqueeze(0)
    global PADDING_MEMS
    _, PADDING_MEMS = txl_model(PADDING_TEXT_TENSOR)



    batch_size = batch_size_dict['pairs']
    num_sents = batch_size_dict['sents']
    if use_wug: 
        batch_size *= 2
        num_sents //= 2

    print('starting all computations at', datetime.now())
    eval_from_file(txl_model, txl_tokenizer, template_name, outfilename, batch_size, num_sents, device=device, use_wug=use_wug, number=number)
    print('completed all computations at', datetime.now())


if __name__ == '__main__':
    main()
