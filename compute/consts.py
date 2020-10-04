# Constants for {model}_compute_sent_probs.py
# Our experiments were run partially on NVIDIA TITAN RTX GPUs with 24GB memory
# and partially with NVIDIA Tesla V100 GPUs with 16GB memory. 
# You need only change the "pairs" number depending on the memory your GPU has. 
# Some of the batch sizes (particularly for the GPT2 ones) are pretty conservative. 

# Template names
SENT_TYPES_SVA = {
    "simple": "SIMPLE",
    "subjrelclause": "SUBJ_REL_CLAUSE",
    "shortvpcoord": "SHORT_VP_COORD",
    "sentcomp": "SENT_COMP",
    "pp": "PP",
    "objrelclausethat": "OBJ_REL_CLAUSE_THAT",
    "objrelclausenothat": "OBJ_REL_CLAUSE_NO_THAT",
    "test": "TEST"
}

SENT_TYPES_ANAPHORA = {
    "simple": "RA_SIMPLE",
    "sentcomp": "RA_SENT_COMP",
    "objrelclausethat": "RA_OBJ_REL_CLAUSE_THAT",
    "objrelclausenothat": "RA_OBJ_REL_CLAUSE_NO_THAT",
}

# Pair batch size - how many "pairs" (S/V) processed in a batch + # of sentences
# included in each pair
BATCH_SIZES_SVA = {
    "simple": {"pairs": 20, "sents": 4}, 
    "subjrelclause": {"pairs": 9, "sents": 8}, # 9x8 for 24GB GPU memory, 5x8 for 16GB GPU memory
    "shortvpcoord": {"pairs": 9, "sents": 8}, # 9x8 for 24GB GPU memory, 5x8 for 16GB GPU memory
    "sentcomp": {"pairs": 9, "sents": 8}, # 9x8 for 24GB GPU memory, 5x8 for 16GB GPU memory
    "pp": {"pairs": 9, "sents": 8}, # 9x8 for 24GB GPU memory, 5x8 for 16GB GPU memory
    "objrelclausethat": {"pairs": 9, "sents": 8}, # 9x8 for 24GB GPU memory, 5x8 for 16GB GPU memory
    "objrelclausenothat": {"pairs": 9, "sents": 8}, # 9x8 for 24GB GPU memory, 5x8 for 16GB GPU memory
}

# note, model size is 4263
BATCH_SIZES_ANAPHORA = {
    "simple": {"pairs": 11, "sents": 4},
    "sentcomp": {"pairs": 5, "sents": 8}, # 9x8 for 24GB GPU memory, 5x8 for 16GB GPU memory
    "objrelclausethat": {"pairs": 5, "sents": 8}, # 9x8 for 24GB GPU memory, 5x8 for 16GB GPU memory
    "objrelclausenothat": {"pairs": 5, "sents": 8}, # 9x8 for 24GB GPU memory, 5x8 for 16GB GPU memory
}

SENT_TYPES = {
    "sv_agreement": SENT_TYPES_SVA,
    "anaphora": SENT_TYPES_ANAPHORA
}

BATCH_SIZES = {
    "sv_agreement": BATCH_SIZES_SVA,
    "anaphora": BATCH_SIZES_ANAPHORA
}

BERT_BATCH_SIZES_SVA = {
    "simple": {"pairs": 2600, "sents": 2}, 
    "subjrelclause": {"pairs": 1000, "sents": 4}, 
    "shortvpcoord": {"pairs": 1200, "sents": 4}, 
    "sentcomp": {"pairs": 1600, "sents": 4}, 
    "pp": {"pairs": 1000, "sents": 4}, 
    "objrelclausethat": {"pairs": 1000, "sents": 4}, 
    "objrelclausenothat": {"pairs": 1100, "sents": 4}, 
}

BERT_BATCH_SIZES_ANAPHORA = {
    "simple": {"pairs": 2600, "sents": 2},
    "sentcomp": {"pairs": 1400, "sents": 4},
    "objrelclausethat": {"pairs": 850, "sents": 4},
    "objrelclausenothat": {"pairs": 900, "sents": 4},
}

BERT_BATCH_SIZES = {
    "sv_agreement": BERT_BATCH_SIZES_SVA,
    "anaphora": BERT_BATCH_SIZES_ANAPHORA
}

MASKED_TYPE_SVA = {
    "simple": '<Verb>', 
    "subjrelclause": '<Verb>', 
    "shortvpcoord": '<Verb>', 
    "sentcomp": '<Verb>', 
    "pp": '<Verb>', 
    "objrelclausethat": '<Verb>', 
    "objrelclausenothat": '<Verb>', 
}

MASKED_TYPE_ANAPHORA = {
    "simple": '<Anaphor>',
    "sentcomp": '<Anaphor>',
    "objrelclausethat": '<Anaphor>',
    "objrelclausenothat": '<Anaphor>',
}

MASKED_TYPE = {
    "sv_agreement": MASKED_TYPE_SVA,
    "anaphora": MASKED_TYPE_ANAPHORA,
}

BERT_MAX_SVA = {
    "simple": 5, 
    "subjrelclause": 10, 
    "shortvpcoord": 8, 
    "sentcomp": 9, 
    "pp": 10, 
    "objrelclausethat": 10, 
    "objrelclausenothat": 9, 
}

BERT_MAX_ANAPHORA = {
    "simple": 7,
    "sentcomp": 11,
    "objrelclausethat": 12,
    "objrelclausenothat": 11,
}

BERT_MAX_TYPE = {
    "sv_agreement": BERT_MAX_SVA,
    "anaphora": BERT_MAX_ANAPHORA,
}

GPT2_BATCH_SIZES_SVA = {
    "simple": {"pairs": 100, "sents": 4},  # on 24GB GPU
    "subjrelclause": {"pairs": 100, "sents": 8},  # on 24GB GPU
    "shortvpcoord": {"pairs": 100, "sents": 8},  # on 24GB GPU
    "sentcomp": {"pairs": 100, "sents": 8},  # on 24GB GPU
    "pp": {"pairs": 100, "sents": 8},  # on 24GB GPU
    "objrelclausethat": {"pairs": 100, "sents": 8},  # on 24GB GPU
    "objrelclausenothat": {"pairs": 100, "sents": 8},  # on 24GB GPU
}

GPT2_BATCH_SIZES_ANAPHORA = {
    "simple": {"pairs": 100, "sents": 4},  # on 24GB GPU
    "sentcomp": {"pairs": 100, "sents": 8},  # on 24GB GPU
    "objrelclausethat": {"pairs": 100, "sents": 8},  # on 24GB GPU
    "objrelclausenothat": {"pairs": 100, "sents": 8},  # on 24GB GPU
}

GPT2_BATCH_SIZES = {
    "sv_agreement": GPT2_BATCH_SIZES_SVA,
    "anaphora": GPT2_BATCH_SIZES_ANAPHORA
}
