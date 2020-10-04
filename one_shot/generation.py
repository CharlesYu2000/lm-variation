# coding=utf-8
# Generation script for GPT2 using fine-tuned models

import os
import sys
import argparse
import pathlib
import copy

from tqdm import trange

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelWithLMHead

torch.set_printoptions(threshold = 25)

# parse arguments
parser  = argparse.ArgumentParser()

parser.add_argument("--model_type", type = str)
parser.add_argument("--model_path", type = str)
parser.add_argument("--contexts", type = str)
parser.add_argument("--gen_outfile", type = str)
parser.add_argument("--dists_outfile", type = str)

args = parser.parse_args()

ABS_PATH = pathlib.Path(__file__).parent.absolute()
MODEL_PATH = os.path.join(ABS_PATH, args.model_path)
#sole_model = args.model_path.split("/")[-1]
#GEN_OUTFILE = os.path.join(ABS_PATH, "gen-" + sole_model + ".txt")
#PRBS_OUTFILE = os.path.join(ABS_PATH, "prbs-" + sole_model + ".txt")

GEN_OUTFILE = os.path.join(ABS_PATH, args.gen_outfile)
PRBS_OUTFILE = os.path.join(ABS_PATH, args.dists_outfile)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelWithLMHead.from_pretrained(MODEL_PATH)
model.eval()
model.to(device)

# Padding text to help Transformer-XL and XLNet with short prompts as proposed by Aman Rusia
# in https://github.com/rusiaaman/XLNet-gen#methodology
# and https://medium.com/@amanrusia/xlnet-speaks-comparison-to-gpt-2-ea1a4e9ba39e
PADDING_TEXT = """ In 1991 , the remains of Russian Tsar Nicholas II and his family
( except for Alexei and Maria ) are discovered . <eos> 
The voice of Nicholas's young son , Tsarevich Alexei Nikolaevich , narrates the
remainder of the story . <eos>  1883 Western Siberia ,
a young Grigori Rasputin is asked by his father and a group of men to perform magic . <eos>
 Rasputin has a vision and denounces one of the men as a horse thief . <eos>  
Although his father initially slaps him for making such an accusation , Rasputin watches as the
man is chased outside and beaten . <eos>  Twenty years later , Rasputin sees a vision of
the Virgin Mary , prompting him to become a priest . <eos>  Rasputin quickly becomes famous ,
with people , even a bishop , begging for his blessing . <eos>"""

# Filters a distribution of logits using top-k filtering
# Parameters:
#   logits - containing logits, shape (batch x vocab)
#   k - the number of top tokens to keep
# Return:
#   tensor of logits of same shape, with values filtered to the top k
def top_k_filtering(logits, k = 1, filter_value = -float('Inf')):
    k = min(k, logits.size(-1)) # check

    if k > 0:
        indices_to_filter = logits < torch.topk(logits, k)[0][..., -1, None]
        logits[indices_to_filter] = filter_value

    return logits

# Examine probability distribution after a given context
def examine_dist(dist, top_k, tokenizer, context_txt, fp):
    # write (topk) probs to file (for inspection) for each dist
    top_logits = torch.topk(dist, top_k, sorted = True)
    top_tokens = [[tokenizer.decode(top_logits[1][0,m,k:k+1]) 
                    for k in range(top_logits[1][0,m].size(0))] 
                        for m in range(dist.size(1))]
    fp.write("Top " + str(top_k) + " probabilities for distribution in " + 
                context_txt + "\n")

    prob_string = ""
    for m in range(dist.size(1)):
        prob_string += "{" + str(2 * m) + ":<16}" + "{" + str(2 * m+1) + ":<16}"

    for j in range(top_k):
        line_tokens = []
        for m in range(dist.size(1)):
            line_tokens.append(repr(top_tokens[m][j]))
            line_tokens.append(str(round(top_logits[0][0,m,j].item(),5)))
        fp.write(prob_string.format(*line_tokens) + "\n")           

# Samples a sequence of specified length with the given model and context
# Parameters:
#   model - the model to sample with
#   length - how long of a sequence to sample
#   context - the context to use when sampling, (batch x 1) [input_ids]
#   top_k - # of top results to write to file for each LM output
#   num_samples - how many sequences to sample
#   fp - file object to write (topk) probs to
# Return:
#   tensor of one-hot encodings of the genrated sequence (batch x length)
def sample_sequence(model, tokenizer, length, context, context_txt, top_k, num_samples, device, fp):
    context = torch.tensor(context, dtype = torch.long, device = device)
    context = context.unsqueeze(0).repeat(num_samples, 1)
    generated = copy.deepcopy(context)
    past = None
    mems = None

    with torch.no_grad():
        for i in trange(length):
            if args.model_type == "gpt2":
                output, past = model(context, past = past)
            elif args.model_type == "txl":
                output, mems = model(context, mems)

            # (batch x sequence_length x vocab_size)
            next_word_logits = output[:, -1, :]
            dist = F.log_softmax(next_word_logits, dim = -1)

            if i == 0:
                examine_dist(dist.unsqueeze(1), top_k, tokenizer, context_txt, fp)

            next_tokens = torch.multinomial(F.softmax(next_word_logits, dim = -1), 
                                            num_samples = 1)
            generated = torch.cat((generated, next_tokens), dim = 1)
            context = next_tokens
    
    return generated


with open(args.contexts, "r") as f:
    gen_contexts = f.read().splitlines()

if args.model_type in ["gpt2", "bert"]:
    context_tokens = [tokenizer.encode(sent) for sent in gen_contexts]
elif args.model_type == "txl":
    # tokenize with padding text
    context_tokens = [PADDING_TEXT + ' ' + context for context in gen_contexts]
    context_tokens = [tokenizer.encode(sent) for sent in context_tokens]

if args.model_type in ["gpt2", "txl"]:
    f_probs = open(PRBS_OUTFILE, "w")
    f_gen = open(GEN_OUTFILE, "w")

    for i in range(len(gen_contexts)):
        print(context_tokens[i])
        model_out = sample_sequence(model = model, tokenizer = tokenizer, length = 7,
            context = context_tokens[i], context_txt = gen_contexts[i], top_k = 50,
            num_samples = 10, device = device, fp = f_probs)


        # Filter out the context itself from the output, (extract only generated txt)
        model_out = model_out[:, len(context_tokens[i]):].tolist()

        print("Context: ", gen_contexts[i])
        f_gen.write("Context: " + gen_contexts[i] + "\n")

        for o in model_out:
            decoded = repr(tokenizer.decode(o))
            print("\t" + decoded)
            f_gen.write("\t" + decoded + "\n")

        print("__________________________________\n")

        f_probs.write("\n")
        f_gen.write("\n")

    f_probs.close()
    f_gen.close()
elif args.model_type == "bert":
    f_probs = open(PRBS_OUTFILE, "w")
    for i in range(len(gen_contexts)):
        input_ids = context_tokens[i]

        try:
            mask_indices = [i for i, input_id in enumerate(input_ids) 
                            if input_id == tokenizer.encode(tokenizer.mask_token)[1]]
        except ValueError:
            print("No [MASK] token in template")
            print(tokenizer.decode(input_ids))
            sys.exit()

        input_ids = torch.tensor([input_ids], device=device)

        out = model(input_ids)
        mask_logits = out[0][:, mask_indices, :]
        mask_dists = F.log_softmax(mask_logits, dim = -1)

        examine_dist(mask_dists, 50, tokenizer, gen_contexts[i], f_probs)
        f_probs.write("\n")

    f_probs.close()