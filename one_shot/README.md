# one_shot

This directory contains results and code for running one/few-shot learning experiments with the GPT2-XL
language model and BERT masked language model.

To obtain results exploring fine-tuned models' predictive distributions and generations,
see `transformer-learning-experiments.csv` and  `run_experiment.sh`. The former
contains specifications for given experiments, while the latter handles the pipeline
for running them. For a given learning experiment, it can be replicated by calling

`bash run_experiment.sh MODEL_NAME EXP_MODEL EXP_TYPE`

with the appopriate model name `MODEL_NAME` listed in the csv for the desired experiment.
`EXP_MODEL` can be 'gpt2', 'bert', or 'both', and `EXP_TYPE` can be 'train', 'gen', or 'both'.
Use 'train' to only fine-tune and save a model, and 'gen' to only run generations for a particular model
(assumes 'train' has already been called). This will locally fine tune and save the appopriate model in the directory `models/NUMBER/CATEGORY/MODEL_NAME/`, and the corresponding results in `results/NUMBER/CATEGORY/MODEL_NAME/`.

To run all the experiments simply use `all` as `MODEL_NAME`, `both` as `EXP_MODEL` and `both` as `EXP_TYPE`. These
experiments use the `gpt2-xl` and `bert-base-uncased` models provided via huggingface.

You may also independently perform your own fine-tuning experiments on GPT2/BERT, by calling

```bash 
bash train_model.sh \
	--model_type=MODEL_TYPE \
	--model_name_or_path=MODEL_NAME \
	--train_data_file=TRAIN_DATA \
	--output_dir=NEW_MODEL_FOLDER \
	--num_train_epochs=EPOCHS
```

where `MODEL_TYPE` is either "gpt2"/"bert", and model_name_or_path can be one of "gpt2-xl"/"bert-base-uncased"
(or another pretrained model in huggingface). This will fine-tune the model on examples in
the `TRAIN_DATA` file, which should have examples separated by lines. The fine-tuned model
will be saved to the path `NEW_MODEL_FOLDER`. Training is done for the specified number of epochs.
More parameters may be specified by calling the `-h` flag.

See `run_language_modeling.py` for more parameters.

If needed, prepare a file with new tokens to be added to the vocabulary at
`NEW_TOKENS_PATH`, with one new token per line. The model and tokenizer will be updated
appropriately to be able to process the new tokens.

To perform generation with a fine-tuned (GPT2) language model, run

```
python generation.py \
 --model_type=MODEL_TYPE \
 --model_path=MODEL_PATH \
 --contexts=CONTEXTS_FILE
 --gen_outfile=GEN_PATH \
 --dists_outfile=MODEL_PATH \
```

where `MODEL_TYPE` is "gpt2" or "bert", `MODEL_PATH` is the folder path to the directory
containing the fine-tuned model, and `CONTEXTS_FILE` is a path to a file that
contains generation contexts, one per line. Generations and distributions will
be saved to the specified paths.

Example:

Train: 
```
bash train_model.sh \
	--model_type=gpt2 \
	--model_name_or_path=gpt2-xl \
	--train_data_file=train_data/test.txt \
	--output_dir=models/test/wug-test \
	--num_train_epochs=1
```

Generation: 
```
python generation.py \
	--model_type=gpt2 \
	--model_path=models/test/wug-test \
	--contexts=contexts/test.txt
	--gen_outfile=results/test_gen.txt \
 	--dists_outfile=results/test_dist.txt \
```

For examples of setting up custom training data, new tokens, and generation contexts, refer to
`train_data` and `contexts`.
