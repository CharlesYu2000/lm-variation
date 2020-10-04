# Compute

This directory contains code for sentence generation and computing language model performance
on grammatical tasks involving these sentences. Each language model works similarly,
though separate scripts are provided to handle the small technical differences between them.

For a given model, the primary script is `{model}_compute_sent_probs.py`, which computes
the model performance scores on a specified set of tasks. The related `{model}_generate_sents.py`
is used within this script to generate the sentences for these tasks, using the associated
word lists and templates in `../sentence_generation`. When calling the compute script,
you should specify the particular sentence type and template to be used. Refer to
`templates.txt` for descriptions and examples of each.

**Example**: Compute model performance with base BERT model on simple subject verb agreement

```bash
# Results saved to various files > ../transformer_evals/bert_sv_agreement/simple/simple_results.csv{counter}
python bert_compute_sent_probs.py -s sv_agreement -t simple
```

For memory and checkpoint purposes the script will produce multiple files under
`../transformer_evals/{model}_{sentence type}/{template}/`. These files will contain
the raw probability differences for each noun with the specified task and its associated templates.
These files can be aggregated for each task afterward to a single large file using `combine_dicts.py`.
The raw probabilities for each task can then be averaged using `convert_numbers.py`.

**Example**

```bash
# aggregate all task result files to single 'FULL' files
python combine_dicts.py -m bert

# > produces aggregated raw scores to ../trasnformer_evals/{model}_{sent type}/{template}.FULL.csv

# compute mean scores for a particular task
python convert_numbers.py \
       -i ../transformer_evals/bert_sv_agreement/subjrelclause.FULL.csv \
       -o ../transformer_evals/bert_sv_agreement/subjrelclause.consolidated.csv
```

The compute scripts may also be used with fine-tuned models; for example, to measure
the performance of models from experiments in `../one_shot/`. To specify a custom model
to evaluate, you may simply add the `-m` option with a path to the model directory. For
instance, to use a fine-tuned transformer model saved via huggingface to `../one_shot/models/fine-tuned-BERT`,
you can call

``` bash
python bert_compute_sent_probs.py -s sv_agreement -t simple -m '../one_shot/models/fine-tuned-BERT'
```

Such results will be saved to `../transformer_evals/fine_tuned/fine-tuned-model-name/`,
under a similar hierarchy as discussed above depending on the category and task.
