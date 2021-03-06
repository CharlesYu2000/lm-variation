# bert_generate_sents.py
# This file functions exactly as `gpt_txl_generate_sents.py` does except it replaces the word we want to 
# predict with a [MASK] token. In our sentences, this just happens to always be the last word, but 
# might not be for other templates. These files are separated for clarity purposes. 

WORDS_DICT_FILENAME = '../sentence_generation/words_dict.json.txt'
NEW_SENTS_FILENAME = '../sentence_generation/new_sents.json.txt'
ALL_SENTS_FILENAME = '../sentence_generation/all_sents.json.txt'
WORDS_DICT_PICKLE_FILENAME = '../sentence_generation/words_dict.pickle'
NEW_SENTS_PICKLE_FILENAME = '../sentence_generation/new_sents.pickle'
ALL_SENTS_PICKLE_FILENAME = '../sentence_generation/all_sents.pickle'

SAMPLE_COUNT = 500
FILE_COUNT = 10


sentence_templates = {
    'SIMPLE': ['The', '<MainNoun>', '[MASK]', '.'], # simple 3 tokens + 1 variable = 5
    'SUBJ_REL_CLAUSE': ['The', '<MainNoun>', 'that', 'liked', 'the', '<Noun>', '[MASK]', '.'], #across subj rel clause   6 tokens + 2 variable (each max len 2) = 10
    'SENT_COMP': ['The', '<Noun>', 'said', 'the', '<MainNoun>', '[MASK]', '.'], # sentential complement     5 tokens + 2 variable (each max len 2) = 9
    'SHORT_VP_COORD': ['The', '<MainNoun>', '<Verb>', 'and', '[MASK]', '.'], # short vp coord, only varying second verb    4 tokens + 2 variable (each max len 2) = 8
    'PP': ['The', '<MainNoun>', 'next', 'to', 'the', '<Noun>', '[MASK]', '.'], # across prep phrase    6 tokens + 2 variable (each max len 2) = 10
    'OBJ_REL_CLAUSE_THAT': ['The', '<MainNoun>', 'that', 'the', '<Noun>', 'liked', '[MASK]', '.'], # across obj rel clause 'that'  6 tokens + 2 variable (each max len 2) = 10
    'OBJ_REL_CLAUSE_NO_THAT': ['The', '<MainNoun>', 'the', '<Noun>', 'liked', '[MASK]', '.'], # across obj rel clause no 'that'  5 tokens + 2 variable (each max len 2) = 9
    'RA_SIMPLE': ['The', '<MainNonGenderedNoun>', '<PastTransVerb>', '[MASK]', '.'], # 3 tokens + 2 variable = 7
    'RA_SENT_COMP': ['The', '<NonGenderedNoun>', 'said', 'the', '<MainNonGenderedNoun>', '<PastTransVerb>', '[MASK]', '.'], # 5 tokens + 3 variable = 11
    'RA_OBJ_REL_CLAUSE_THAT': ['The', '<MainNonGenderedNoun>', 'that', 'the', '<NonGenderedNoun>', 'liked', '<PastTransVerb>', '[MASK]', '.'], # 6 tokens + 3 variable = 12
    'RA_OBJ_REL_CLAUSE_NO_THAT': ['The', '<MainNonGenderedNoun>', 'the', '<NonGenderedNoun>', 'liked', '<PastTransVerb>', '[MASK]', '.'], # 5 tokens + 3 variable = 11
}

def load_words_dict_from_pickle(filename=None): 
    from collections import defaultdict
    try:
        import cPickle as pickle
    except: 
        import pickle
    if filename is None: 
        words_dict = {'<Verb>': defaultdict(list), '<Noun>': defaultdict(list)}
    else: 
        with open(filename, 'rb') as f: 
            words_dict = pickle.load(f)
    return words_dict

def load_words_dict_from_json(filename=None): 
    from collections import defaultdict
    import json
    if filename is None: 
        words_dict = {'<Verb>': defaultdict(list), '<Noun>': defaultdict(list)}
    else: 
        with open(filename, 'r', encoding='utf8') as f:
            words_dict = json.load(f)
    return words_dict

def load_words_list_from_json(filename=None): 
    import json
    if filename is None: 
        words_list = None
    else: 
        with open(filename, 'r', encoding='utf8') as f: 
            words_list = json.load(f)
    return words_list

def load_nouns_list_from_json(filename='Noun_list.json.txt'):
    return load_words_list_from_json(filename=filename)

def load_verbs_list_from_json(filename='Verb_list.json.txt'): 
    return load_words_list_from_json(filename=filename)

def load_trans_verbs_list_from_json(filename='TransVerb_list.json.txt'): 
    return load_words_dict_from_json(filename=filename)

def load_past_trans_verbs_list_from_json(filename='PastTransVerb_list.json.txt'): 
    return load_words_dict_from_json(filename=filename)

def load_non_gendered_nouns_list_from_json(filename='NonGenderedNoun_list.json.txt'): 
    return load_words_dict_from_json(filename=filename)

def load_anaphors_list_from_json(filename='Anaphor_list.json.txt'): 
    return load_words_list_from_json(filename=filename)
    
def save_to_file(all_sents, new_sents, words_dict, prefix=''): 
    save_all_formats(words_dict, {'pickle': WORDS_DICT_PICKLE_FILENAME, 'json': WORDS_DICT_FILENAME})
    save_all_formats(new_sents, {'pickle': prefix + NEW_SENTS_PICKLE_FILENAME, 'json': prefix + NEW_SENTS_FILENAME})
    save_all_formats(all_sents, {'pickle': prefix + ALL_SENTS_PICKLE_FILENAME, 'json': prefix + ALL_SENTS_FILENAME})

def save_all_formats(data, filename_dict): 
    pickle_save(data, filename_dict['pickle'])
    json_save(data, filename_dict['json'])

def pickle_save(data, filename): 
    try:
        import cPickle as pickle
    except: 
        import pickle
    with open(filename, 'wb') as f: 
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def json_save(data, filename): 
    import json
    with open(filename, 'w', encoding='utf8') as f:
        json.dump(data, f, indent=2, separators=(',', ': '))

def split_dict(new_sents, dict_count): 
    from itertools import islice
    each_size = len(new_sents)//dict_count
    dict_iterator = iter(new_sents)
    for i in range(0, len(new_sents), each_size): 
        yield {k: new_sents[k] for k in islice(dict_iterator, each_size)}


def fill_template_simple(all_sents, new_sents, template, words_list, words_list_dict=None, words_dict=load_words_dict_from_json(filename=WORDS_DICT_FILENAME), start=0): 
    import random
    from collections import defaultdict
    from datetime import datetime
    print('filling simple template')
    rand_seed = 541
    random.seed(rand_seed)
    print('begin sampling simple sentences using seed %d' % rand_seed)
    count = 0
    if words_list_dict is None:
        main_noun_list = list(words_dict['<MainNoun>'].keys())
        noun_list = list(words_dict['<Noun>'].keys())
    else: 
        main_noun_list = [x[0] for x in words_list_dict['<MainNoun>']]
        noun_list = [x[0] for x in words_list_dict['<Noun>']]
    
    print('done sampling simple sentences')
    print('begin generating simple sentences')
    
    for main_noun in main_noun_list: 
        for main_noun_form in words_dict['<MainNoun>'][main_noun]: 
            template[1] = main_noun_form

            count += 1 
            new_sents[str((main_noun,))].append(template.copy())
            if count%1048576 == 0: 
                print("Done with %d" % count)
                print('at', datetime.now())
    
    print('done generating simple sentences') 

def fill_template_subjrelclause(all_sents, new_sents, template, words_list, words_list_dict=None, words_dict=load_words_dict_from_json(filename=WORDS_DICT_FILENAME)): 
    import random
    from collections import defaultdict
    from datetime import datetime
    print('filling across a subject relative clause template')
    rand_seed = 118
    random.seed(rand_seed)
    print('begin sampling across a subject relative clause sentences using seed %d' % rand_seed)
    count = 0
    subsamp_words_list = defaultdict(list)
    if words_list_dict is None:
        main_noun_list = list(words_dict['<MainNoun>'].keys())
        noun_list = list(words_dict['<Noun>'].keys())
    else: 
        main_noun_list = [x[0] for x in words_list_dict['<MainNoun>']]
        noun_list = [x[0] for x in words_list_dict['<Noun>']]

    for noun in main_noun_list: 
        subsamplist = random.sample(noun_list, min(len(noun_list),SAMPLE_COUNT))
        for samp_noun in subsamplist: 
            subsamp_words_list[noun].append((samp_noun,))
            
    
    print('done sampling across a subject relative clause sentences')
    print('begin generating across a subject relative clause sentences')
    
    for main_noun in main_noun_list: 
        for main_noun_form in words_dict['<MainNoun>'][main_noun]: 
            template[1] = main_noun_form

            for second_noun, in subsamp_words_list[main_noun]: 
                for second_noun_form in words_dict['<Noun>'][second_noun]: 
                    template[5] = second_noun_form

                    count += 1 
                    new_sents[str((main_noun, second_noun))].append(template.copy())
                    if count%1048576 == 0: 
                        print("Done with %d" % count)
                        print('at', datetime.now())
    
    print('done generating across a subject relative clause sentences')   


def fill_template_sent_comp(all_sents, new_sents, template, words_list, words_list_dict=None, words_dict=load_words_dict_from_json(filename=WORDS_DICT_FILENAME)): 
    import random
    from collections import defaultdict
    from datetime import datetime
    print('filling sentential complement template')
    rand_seed = 713
    random.seed(rand_seed)
    print('begin sampling sentential complement sentences using seed %d' % rand_seed)
    count = 0
    subsamp_words_list = defaultdict(list)
    if words_list_dict is None:
        main_noun_list = list(words_dict['<MainNoun>'].keys())
        noun_list = list(words_dict['<Noun>'].keys())
    else: 
        main_noun_list = [x[0] for x in words_list_dict['<MainNoun>']]
        noun_list = [x[0] for x in words_list_dict['<Noun>']]

    for noun in main_noun_list: 
        subsamplist = random.sample(noun_list, min(len(noun_list),SAMPLE_COUNT))
        for samp_noun in subsamplist: 
            subsamp_words_list[noun].append((samp_noun,))
    
    print('done sampling sentential complement sentences')
    print('begin generating sentential complement sentences')
    
    for main_noun in main_noun_list: 
        for main_noun_form in words_dict['<MainNoun>'][main_noun]: 
            template[4] = main_noun_form

            for second_noun, in subsamp_words_list[main_noun]: 
                for second_noun_form in words_dict['<Noun>'][second_noun]: 
                    template[1] = second_noun_form

                    count += 1 
                    new_sents[str((main_noun, second_noun))].append(template.copy())
                    if count%1048576 == 0: 
                        print("Done with %d" % count)
                        print('at', datetime.now())
    print('done generating sentential complement sentences')        


def fill_template_short_vp_coord(all_sents, new_sents, template, words_list, words_list_dict=None, words_dict=load_words_dict_from_json(filename=WORDS_DICT_FILENAME)): 
    import random
    from collections import defaultdict
    from datetime import datetime
    print('filling short VP coordination template')
    rand_seed = 621
    random.seed(rand_seed)
    print('begin sampling short VP coordination sentences using seed %d' % rand_seed)
    count = 0
    subsamp_words_list = defaultdict(list)
    if words_list_dict is None:
        main_noun_list = list(words_dict['<MainNoun>'].keys())
        noun_list = list(words_dict['<Noun>'].keys())
        verb_list = list(words_dict['<Verb>'].keys())
    else: 
        main_noun_list = [x[0] for x in words_list_dict['<MainNoun>']]
        noun_list = [x[0] for x in words_list_dict['<Noun>']]
        verb_list = [x[0] for x in words_list_dict['<Verb>']]
    
    for noun in main_noun_list: 
        subsamplist = random.sample(verb_list, min(len(verb_list),SAMPLE_COUNT))
        for samp_verb in subsamplist: 
            subsamp_words_list[noun].append((samp_verb,))
    
    print('done sampling short VP coordination sentences')
    print('begin generating short VP coordination sentences')
    
    for main_noun in main_noun_list: 
        for main_noun_form in words_dict['<MainNoun>'][main_noun]: 
            template[1] = main_noun_form

            # for first_verb, in subsamp_words_list[main_noun]: 
            for first_verb in verb_list: 
                for first_verb_form in words_dict['<Verb>'][first_verb]: 
                    template[2] = first_verb_form

                    count += 1 
                    new_sents[str((main_noun, first_verb))].append(template.copy())
                    if count%1048576 == 0: 
                        print("Done with %d" % count)
                        print('at', datetime.now())
    print('done generating short VP coordination sentences')          


def fill_template_prep_phrase(all_sents, new_sents, template, words_list, words_list_dict=None, words_dict=load_words_dict_from_json(filename=WORDS_DICT_FILENAME)): 
    import random
    from collections import defaultdict
    from datetime import datetime
    print('filling across a prepositional phrase template')
    rand_seed = 430
    random.seed(rand_seed)
    print('begin sampling across a prepositional phrase sentences using seed %d' % rand_seed)
    count = 0
    subsamp_words_dict = defaultdict(set)
    subsamp_words_list = defaultdict(list)
    if words_list_dict is None:
        main_noun_list = list(words_dict['<MainNoun>'].keys())
        noun_list = list(words_dict['<Noun>'].keys())
    else: 
        main_noun_list = [x[0] for x in words_list_dict['<MainNoun>']]
        noun_list = [x[0] for x in words_list_dict['<Noun>']]

    for noun in main_noun_list: 
        subsamplist = random.sample(noun_list, min(len(noun_list),SAMPLE_COUNT))
        for samp_noun in subsamplist: 
            subsamp_words_list[noun].append((samp_noun,))
    
    print('done sampling across a prepositional phrase sentences')
    print('begin generating across a prepositional phrase sentences')
    
    for main_noun in main_noun_list: 
        for main_noun_form in words_dict['<MainNoun>'][main_noun]: 
            template[1] = main_noun_form

            for second_noun, in subsamp_words_list[main_noun]: 
                for second_noun_form in words_dict['<Noun>'][second_noun]: 
                    template[5] = second_noun_form

                    count += 1 
                    new_sents[str((main_noun, second_noun))].append(template.copy())
                    if count%1048576 == 0: 
                        print("Done with %d" % count)
                        print('at', datetime.now())
    print('done generating across a prepositional phrase sentences')     

def fill_template_objrelclause_that(all_sents, new_sents, template, words_list, words_list_dict=None, words_dict=load_words_dict_from_json(filename=WORDS_DICT_FILENAME)): 
    import random
    from collections import defaultdict
    from datetime import datetime
    print('filling across an object relative clause WITH \'that\' template')
    rand_seed = 1008
    random.seed(rand_seed)
    print('begin sampling across an object relative clause WITH \'that\' sentences using seed %d' % rand_seed)
    count = 0
    subsamp_words_dict = defaultdict(set)
    subsamp_words_list = defaultdict(list)
    if words_list_dict is None:
        main_noun_list = list(words_dict['<MainNoun>'].keys())
        noun_list = list(words_dict['<Noun>'].keys())
    else: 
        main_noun_list = [x[0] for x in words_list_dict['<MainNoun>']]
        noun_list = [x[0] for x in words_list_dict['<Noun>']]

    for noun in main_noun_list: 
        subsamplist = random.sample(noun_list, min(len(noun_list),SAMPLE_COUNT))
        for samp_noun in subsamplist: 
            subsamp_words_list[noun].append((samp_noun,))            
    
    print('done sampling across an object relative clause WITH \'that\' sentences')
    print('begin generating across an object relative clause WITH \'that\' sentences')
    
    for main_noun in main_noun_list: 
        for main_noun_form in words_dict['<MainNoun>'][main_noun]: 
            template[1] = main_noun_form

            for second_noun, in subsamp_words_list[main_noun]: 
                for second_noun_form in words_dict['<Noun>'][second_noun]: 
                    template[4] = second_noun_form

                    count += 1 
                    new_sents[str((main_noun, second_noun))].append(template.copy())
                    if count%1048576 == 0: 
                        print("Done with %d" % count)
                        print('at', datetime.now())
    print('done generating across an object relative clause WITH \'that\' sentences')     

def fill_template_objrelclause_nothat(all_sents, new_sents, template, words_list, words_list_dict=None, words_dict=load_words_dict_from_json(filename=WORDS_DICT_FILENAME)): 
    import random
    from collections import defaultdict
    from datetime import datetime
    print('filling across an object relative clause WITHOUT \'that\' template')
    rand_seed = 601
    random.seed(rand_seed)
    print('begin sampling across an object relative clause WITHOUT \'that\' sentences using seed %d' % rand_seed)
    count = 0
    subsamp_words_dict = defaultdict(set)
    subsamp_words_list = defaultdict(list)
    if words_list_dict is None:
        main_noun_list = list(words_dict['<MainNoun>'].keys())
        noun_list = list(words_dict['<Noun>'].keys())
    else: 
        main_noun_list = [x[0] for x in words_list_dict['<MainNoun>']]
        noun_list = [x[0] for x in words_list_dict['<Noun>']]

    for noun in main_noun_list: 
        subsamplist = random.sample(noun_list, min(len(noun_list),SAMPLE_COUNT))
        for samp_noun in subsamplist: 
            subsamp_words_list[noun].append((samp_noun,))    
    
    print('done sampling across an object relative clause WITHOUT \'that\' sentences')
    print('begin generating across an object relative clause WITHOUT \'that\' sentences')
    
    for main_noun in main_noun_list: 
        for main_noun_form in words_dict['<MainNoun>'][main_noun]: 
            template[1] = main_noun_form

            for second_noun, in subsamp_words_list[main_noun]: 
                for second_noun_form in words_dict['<Noun>'][second_noun]: 
                    template[3] = second_noun_form

                    count += 1 
                    new_sents[str((main_noun, second_noun))].append(template.copy())
                    if count%1048576 == 0: 
                        print("Done with %d" % count)
                        print('at', datetime.now())
    print('done generating across an object relative clause WITHOUT \'that\' sentences')  

def fill_template_ra_simple(all_sents, new_sents, template, words_list, words_list_dict=None, words_dict=load_words_dict_from_json(filename=WORDS_DICT_FILENAME)): 
    import random
    from collections import defaultdict
    from datetime import datetime
    print('filling reflexive anaphora simple template')
    rand_seed = 22
    random.seed(rand_seed)
    print('begin sampling reflexive anaphora simple sentences using seed %d' % rand_seed)
    count = 0
    subsamp_words_dict = defaultdict(set)
    subsamp_words_list = defaultdict(list)
    if words_list_dict is None:
        main_noun_list = list(words_dict['<MainNonGenderedNoun>'].keys())
        noun_list = list(words_dict['<NonGenderedNoun>'].keys())
        verb_list = list(words_dict['<PastTransVerb>'].keys())
    else: 
        main_noun_list = [x[0] for x in words_list_dict['<MainNonGenderedNoun>']]
        noun_list = [x[0] for x in words_list_dict['<NonGenderedNoun>']]
        verb_list = [x[0] for x in words_list_dict['<PastTransVerb>']]

    for noun in main_noun_list: 
        comb_count = len(verb_list)
        subsampinds = random.sample(range(comb_count), min(comb_count,SAMPLE_COUNT))
        for samp_ind in subsampinds: 
            samp_verb = verb_list[samp_ind]
            subsamp_words_list[noun].append((samp_verb,))
    
    print('done sampling reflexive anaphora simple sentences')
    print('begin generating reflexive anaphora simple sentences')
    
    for main_noun in main_noun_list: 
        for main_noun_form in words_dict['<MainNonGenderedNoun>'][main_noun]: 
            template[1] = main_noun_form

            for main_verb, in subsamp_words_list[main_noun]: 
                for verb_form in words_dict['<PastTransVerb>'][main_verb]: 
                    template[2] = verb_form

                    count += 1 
                    new_sents[str((main_noun, main_verb))].append(template.copy())
                    if count%1048576 == 0: 
                        print("Done with %d" % count)
                        print('at', datetime.now())
    print('done generating reflexive anaphora simple sentences')    
 
def fill_template_ra_sentcomp(all_sents, new_sents, template, words_list, words_list_dict=None, words_dict=load_words_dict_from_json(filename=WORDS_DICT_FILENAME)): 
    import random
    from collections import defaultdict
    from datetime import datetime
    print('filling reflexive anaphora sentcomp template')
    rand_seed = 19
    random.seed(rand_seed)
    print('begin sampling reflexive anaphora sentcomp sentences using seed %d' % rand_seed)
    count = 0
    subsamp_words_dict = defaultdict(set)
    subsamp_words_list = defaultdict(list)
    if words_list_dict is None:
        main_noun_list = list(words_dict['<MainNonGenderedNoun>'].keys())
        noun_list = list(words_dict['<NonGenderedNoun>'].keys())
        verb_list = list(words_dict['<PastTransVerb>'].keys())
        anaphor_list = list(words_dict['<Anaphor>'].keys())
    else: 
        main_noun_list = [x[0] for x in words_list_dict['<MainNonGenderedNoun>']]
        noun_list = [x[0] for x in words_list_dict['<NonGenderedNoun>']]
        verb_list = [x[0] for x in words_list_dict['<PastTransVerb>']]
        anaphor_list = [x[0] for x in words_list_dict['<Anaphor>']]

    for noun in main_noun_list: 
        comb_count = len(noun_list)*len(verb_list)
        subsampinds = random.sample(range(comb_count), min(comb_count,SAMPLE_COUNT))
        for samp_ind in subsampinds: 
            samp_noun = noun_list[samp_ind//len(verb_list)]
            samp_verb = verb_list[samp_ind % len(verb_list)]
            subsamp_words_list[noun].append((samp_noun, samp_verb))
    
    print('done sampling reflexive anaphora sentcomp sentences')
    print('begin generating reflexive anaphora sentcomp sentences')
    
    for main_noun in main_noun_list: 
        for main_noun_form in words_dict['<MainNonGenderedNoun>'][main_noun]: 
            template[4] = main_noun_form

            for second_noun, main_verb in subsamp_words_list[main_noun]: 
                for noun_form in words_dict['<NonGenderedNoun>'][second_noun]:
                    template[1] = noun_form

                    for verb_form in words_dict['<PastTransVerb>'][main_verb]: 
                        template[5] = verb_form
 
                        count += 1 
                        new_sents[str((main_noun, second_noun, main_verb))].append(template.copy())
                        if count%1048576 == 0: 
                            print("Done with %d" % count)
                            print('at', datetime.now())
    print('done generating reflexive anaphora sentcomp sentences')     
 
def fill_template_ra_objrelclause_that(all_sents, new_sents, template, words_list, words_list_dict=None, words_dict=load_words_dict_from_json(filename=WORDS_DICT_FILENAME)): 
    import random
    from collections import defaultdict
    from datetime import datetime
    print('filling reflexive anaphora rel clause template')
    rand_seed = 551
    random.seed(rand_seed)
    print('begin sampling reflexive anaphora rel clause sentences using seed %d' % rand_seed)
    count = 0
    subsamp_words_dict = defaultdict(set)
    subsamp_words_list = defaultdict(list)
    if words_list_dict is None:
        main_noun_list = list(words_dict['<MainNonGenderedNoun>'].keys())
        noun_list = list(words_dict['<NonGenderedNoun>'].keys())
        verb_list = list(words_dict['<PastTransVerb>'].keys())
        anaphor_list = list(words_dict['<Anaphor>'].keys())
    else: 
        main_noun_list = [x[0] for x in words_list_dict['<MainNonGenderedNoun>']]
        noun_list = [x[0] for x in words_list_dict['<NonGenderedNoun>']]
        verb_list = [x[0] for x in words_list_dict['<PastTransVerb>']]
        anaphor_list = [x[0] for x in words_list_dict['<Anaphor>']]

    for noun in main_noun_list: 
        comb_count = len(noun_list)*len(verb_list)
        subsampinds = random.sample(range(comb_count), min(comb_count,SAMPLE_COUNT))
        for samp_ind in subsampinds: 
            samp_noun = noun_list[samp_ind//len(verb_list)]
            samp_verb = verb_list[samp_ind % len(verb_list)]
            subsamp_words_list[noun].append((samp_noun, samp_verb))            
    
    print('done sampling reflexive anaphora rel clause sentences')
    print('begin generating reflexive anaphora rel clause sentences')
    
    for main_noun in main_noun_list: 
        for main_noun_form in words_dict['<MainNonGenderedNoun>'][main_noun]: 
            template[1] = main_noun_form

            for second_noun, main_verb in subsamp_words_list[main_noun]: 
                for noun_form in words_dict['<NonGenderedNoun>'][second_noun]:
                    template[4] = noun_form

                    for verb_form in words_dict['<PastTransVerb>'][main_verb]: 
                        template[6] = verb_form

                        count += 1 
                        new_sents[str((main_noun, second_noun, main_verb))].append(template.copy())
                        if count%1048576 == 0: 
                            print("Done with %d" % count)
                            print('at', datetime.now())
    print('done generating reflexive anaphora rel clause sentences')     
 
def fill_template_ra_objrelclause_nothat(all_sents, new_sents, template, words_list, words_list_dict=None, words_dict=load_words_dict_from_json(filename=WORDS_DICT_FILENAME)): 
    import random
    from collections import defaultdict
    from datetime import datetime
    print('filling reflexive anaphora rel clause template')
    rand_seed = 1203
    random.seed(rand_seed)
    print('begin sampling reflexive anaphora rel clause sentences using seed %d' % rand_seed)
    count = 0
    subsamp_words_dict = defaultdict(set)
    subsamp_words_list = defaultdict(list)
    if words_list_dict is None:
        main_noun_list = list(words_dict['<MainNonGenderedNoun>'].keys())
        noun_list = list(words_dict['<NonGenderedNoun>'].keys())
        verb_list = list(words_dict['<PastTransVerb>'].keys())
        anaphor_list = list(words_dict['<Anaphor>'].keys())
    else: 
        main_noun_list = [x[0] for x in words_list_dict['<MainNonGenderedNoun>']]
        noun_list = [x[0] for x in words_list_dict['<NonGenderedNoun>']]
        verb_list = [x[0] for x in words_list_dict['<PastTransVerb>']]
        anaphor_list = [x[0] for x in words_list_dict['<Anaphor>']]

    for noun in main_noun_list: 
        comb_count = len(noun_list)*len(verb_list)
        subsampinds = random.sample(range(comb_count), min(comb_count,SAMPLE_COUNT))
        for samp_ind in subsampinds: 
            samp_noun = noun_list[samp_ind//len(verb_list)]
            samp_verb = verb_list[samp_ind % len(verb_list)]
            subsamp_words_list[noun].append((samp_noun, samp_verb))
            
    
    print('done sampling reflexive anaphora rel clause sentences')
    print('begin generating reflexive anaphora rel clause sentences')
    
    for main_noun in main_noun_list: 
        for main_noun_form in words_dict['<MainNonGenderedNoun>'][main_noun]: 
            template[1] = main_noun_form

            for second_noun, main_verb in subsamp_words_list[main_noun]: 
                for noun_form in words_dict['<NonGenderedNoun>'][second_noun]:
                    template[3] = noun_form

                    for verb_form in words_dict['<PastTransVerb>'][main_verb]: 
                        template[5] = verb_form

                        count += 1 
                        new_sents[str((main_noun, second_noun, main_verb))].append(template.copy())
                        if count%1048576 == 0: 
                            print("Done with %d" % count)
                            print('at', datetime.now())
    print('done generating reflexive anaphora rel clause sentences')     

def fill_templates(template, fill_template_helper, words_list_dict=None, words_dict=load_words_dict_from_json(WORDS_DICT_FILENAME), prefix='', save_to_file=True): 
    from collections import defaultdict
    import time
    from datetime import datetime
    import os

    total_start_time = time.monotonic()
    start_time = time.monotonic()
    try: 
        with open(prefix + ALL_SENTS_PICKLE_FILENAME, 'rb') as f: 
            all_sents = pickle.load(f)
    except: 
        all_sents = {}
    all_sents = defaultdict(list, all_sents)
    elapsed_time = time.monotonic()-start_time
    print('loaded all current sentences in %d seconds' % elapsed_time)
    new_sents = defaultdict(list)
    print('filling template %s' % str(template))
    print('Started at', datetime.now())
    start_time = time.monotonic()
    fill_template_helper(all_sents, new_sents, template, list(), words_list_dict=words_list_dict, words_dict=words_dict)
    elapsed_time = time.monotonic()-start_time
    print('done filling template (example: %s) in %d seconds' % (str(template), elapsed_time))
    all_sents.update(new_sents)
    if save_to_file: 
        print('saving sentences to file')
        if '/' in prefix: 
            dir_name = './' + prefix.split('/')[0]
            if not os.path.isdir(dir_name): 
                print('creating subdirectory %s' % dir_name)
                os.mkdir(dir_name)

        # save_all_formats(all_sents, {'pickle': prefix + ALL_SENTS_PICKLE_FILENAME, 'json': prefix + ALL_SENTS_FILENAME})
        new_sents_list = []

        print('splitting into %d file(s)' % FILE_COUNT)

        for new_dict in split_dict(new_sents, FILE_COUNT): 
            new_sents_list.append(new_dict)

        for i in range(FILE_COUNT): 
            print('saving file %d' % i)
            json_save(new_sents_list[i], prefix + NEW_SENTS_FILENAME + '.' + str(i))

        print('done saving sentences to file')
    total_elapsed_time = time.monotonic()-total_start_time
    print('finished filling this templates in %d seconds' % total_elapsed_time)
    print('generated %d sentences' % len(new_sents))


    with open('new_sents_file.pickle', 'wb') as f: 
        import pickle
        pickle.dump(new_sents, f, protocol=pickle.HIGHEST_PROTOCOL)

    return new_sents

fill_template_helpers = {
    'SIMPLE': fill_template_simple, # simple
    'SUBJ_REL_CLAUSE': fill_template_subjrelclause, #across subj rel clause
    'SENT_COMP': fill_template_sent_comp, # sentential complement
    'SHORT_VP_COORD': fill_template_short_vp_coord, # short vp coord, only varying second verb
    'PP': fill_template_prep_phrase, # across prep phrase
    'OBJ_REL_CLAUSE_THAT': fill_template_objrelclause_that, # across obj rel clause 'that'
    'OBJ_REL_CLAUSE_NO_THAT': fill_template_objrelclause_nothat, # across obj rel clause no 'that'
    'RA_SIMPLE': fill_template_ra_simple, 
    'RA_SENT_COMP': fill_template_ra_sentcomp, 
    'RA_OBJ_REL_CLAUSE_THAT': fill_template_ra_objrelclause_that, 
    'RA_OBJ_REL_CLAUSE_NO_THAT': fill_template_ra_objrelclause_nothat,
}

prefixes = {
    'SIMPLE': 'simple/simple_', # simple
    'SUBJ_REL_CLAUSE': 'subjrelclause/subjrelclause_', #across subj rel clause
    'SENT_COMP': 'sentcomp/sentcomp_', # sentential complement
    'SHORT_VP_COORD': 'shortvpcoord/shortvpcoord_', # short vp coord, only varying second verb
    'PP': 'pp/pp_', # across prep phrase
    'OBJ_REL_CLAUSE_THAT': 'objrelclausethat/objrelclausethat_', # across obj rel clause 'that'
    'OBJ_REL_CLAUSE_NO_THAT': 'objrelclausenothat/objrelclausenothat_', # across obj rel clause no 'that'
    'RA_SIMPLE': 'ra_simple/ra_simple_', 
    'RA_SENT_COMP': 'ra_sentcomp/ra_sentcomp_', 
    'RA_OBJ_REL_CLAUSE_THAT': 'ra_objrelclausethat/ra_objrelclausethat_', 
    'RA_OBJ_REL_CLAUSE_NO_THAT': 'ra_objrelclausenothat/ra_objrelclausenothat_',
}

if __name__ == '__main__': 
    print('starting main method')
    words_dict = load_words_dict_from_json(WORDS_DICT_FILENAME)
    words_list_dict = {
        # '<Noun>': load_nouns_list_from_json(), 
        # '<Verb>': load_verbs_list_from_json(), 
        '<NonGenderedNoun>': load_non_gendered_nouns_list_from_json(), 
        '<PastTransVerb>': load_past_trans_verbs_list_from_json(), 
        '<Anaphor>': load_anaphors_list_from_json(), 
    }
    # fill_templates(sentence_templates['SUBJ_REL_CLAUSE'], fill_template_helpers['SUBJ_REL_CLAUSE'], words_list_dict=words_list_dict, words_dict=words_dict, prefix=prefixes['SUBJ_REL_CLAUSE'])
    # fill_templates(sentence_templates['SENT_COMP'], fill_template_helpers['SENT_COMP'], words_list_dict=words_list_dict, words_dict=words_dict, prefix=prefixes['SENT_COMP'])
    # fill_templates(sentence_templates['SHORT_VP_COORD'], fill_template_helpers['SHORT_VP_COORD'], words_list_dict=words_list_dict, words_dict=words_dict, prefix=prefixes['SHORT_VP_COORD'])
    # fill_templates(sentence_templates['PP'], fill_template_helpers['PP'], words_list_dict=words_list_dict, words_dict=words_dict, prefix=prefixes['PP'])
    # fill_templates(sentence_templates['OBJ_REL_CLAUSE_THAT'], fill_template_helpers['OBJ_REL_CLAUSE_THAT'], words_list_dict=words_list_dict, words_dict=words_dict, prefix=prefixes['OBJ_REL_CLAUSE_THAT'])
    # fill_templates(sentence_templates['OBJ_REL_CLAUSE_NO_THAT'], fill_template_helpers['OBJ_REL_CLAUSE_NO_THAT'], words_list_dict=words_list_dict, words_dict=words_dict, prefix=prefixes['OBJ_REL_CLAUSE_NO_THAT'])
    fill_templates(sentence_templates['RA_SIMPLE'], fill_template_helpers['RA_SIMPLE'], words_list_dict=words_list_dict, words_dict=words_dict, prefix=prefixes['RA_SIMPLE'])
    fill_templates(sentence_templates['RA_SENT_COMP'], fill_template_helpers['RA_SENT_COMP'], words_list_dict=words_list_dict, words_dict=words_dict, prefix=prefixes['RA_SENT_COMP'])
    fill_templates(sentence_templates['RA_OBJ_REL_CLAUSE_THAT'], fill_template_helpers['RA_OBJ_REL_CLAUSE_THAT'], words_list_dict=words_list_dict, words_dict=words_dict, prefix=prefixes['RA_OBJ_REL_CLAUSE_THAT'])
    fill_templates(sentence_templates['RA_OBJ_REL_CLAUSE_NO_THAT'], fill_template_helpers['RA_OBJ_REL_CLAUSE_NO_THAT'], words_list_dict=words_list_dict, words_dict=words_dict, prefix=prefixes['RA_OBJ_REL_CLAUSE_NO_THAT'])
