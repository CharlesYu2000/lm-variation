#!/bin/bash

MODEL_NAME=$1
EXP_MODEL=$2
TASK=$3

BERT_EPOCHS=4
GPT2_EPOCHS=2

if [[ $MODEL_NAME == "all" ]]; then
    echo "Running ALL experiments..."
fi

# Get updated csv
bash get_csv.sh

# Function to fine-tune gpt2-xl model + run generations
#   $1 = model name, $2 = path to training data,
#   $3 = output directory, $4 = # of training epochs,
#   $5 = results output directory, $6 = sg or pl
#   $7 = model type, $8 = model name, $9 = token file
train_model() {    
    if [[ "$TASK" == "train" ]] || [[ "$TASK" == "both" ]] ; then    
        # Fine tune the gpt2-xl model
        echo "Fine-tuning $1-$8 on the examples in $2"

        ADDL_ARGS=()

        if [[ "$mname" == "wug-base" ]] || [[ "$mname" == "wuz-base" ]]; then
            ADDL_ARGS+=(--base_model)
            ITER=30
        else
            ITER=1
        fi

        if [[ "$7" == "bert" ]]; then 
            ADDL_ARGS+=(--mlm)
        fi

        OUT_DIR=$3/$1-$8

        while [[ $ITER -ge 1  ]]; do
            if [[ "$mname" == "wug-base" ]] || [[ "$mname" == "wuz-base" ]]; then
                ADDL_ARGS+=(--output_dir=$OUT_DIR-$ITER)
            else
                ADDL_ARGS+=(--output_dir=$OUT_DIR)
            fi

            bash train_model.sh \
                --model_type=$7 \
                --model_name_or_path=$8 \
                --train_data_file=$2 \
                --num_train_epochs=$4 \
                --new_tokens_path=train_data/wugz/$9 \
                --seed=$ITER \
                ${ADDL_ARGS[@]}
            let ITER-=1
        done
    fi

    # Run generations/examine distributions on the fine-tuned model  
    if [[ "$TASK" == "gen" ]] || [[ "$TASK" == "both" ]] ; then  
        echo "Examining $1-$8"

        python generation.py \
            --model_type=$7 \
            --model_path=$3/$1-$8 \
            --contexts=$6 \
            --gen_outfile=$5/gen-$1-$8.txt \
            --dists_outfile=$5/dists-$1-$8.txt \

        echo "Results saved to $5/"
    fi
}

# Setup directory structure for models/results, obtain training
# data file, experiment category and number
echo "Setting up models/ and results/ directories"
EXP_CSV=transformer_learning_experiments.csv
IFS=","
while read mname num cat stim distributive collective
do
    if [[ $mname == "model_name" ]]; then
        continue
    fi

    mkdir -p models/$num/$cat/
    mkdir -p results/$num/$cat/$mname

    if [[ "$mname" == "$MODEL_NAME" ]] || [[ "$MODEL_NAME" == "all" ]]; then
        TRAIN_DATA_FILE=train_data/wugz/$num/$cat/$mname.txt

        MODEL_OUTPUT_DIR=models/$num/$cat
        RESULTS_OUTPUT_DIR=results/$num/$cat/$mname

        if [[ "$num" == "sg" ]]; then
            TOKEN_FILE=wug-token.txt
        elif [[ "$num" == "pl" ]]; then
            TOKEN_FILE=wuz-token.txt
        fi

        if [[ "$EXP_MODEL" == "gpt2" ]] || [[ "$EXP_MODEL" == "both" ]]; then
            # exp on gpt2
            train_model $mname $TRAIN_DATA_FILE $MODEL_OUTPUT_DIR \
                $GPT2_EPOCHS $RESULTS_OUTPUT_DIR contexts/wugz-$num.txt \
                gpt2 gpt2-xl $TOKEN_FILE
        fi
        if [[ "$EXP_MODEL" == "bert" ]] || [[ "$EXP_MODEL" == "both" ]]; then
            # exp on bert
            train_model $mname $TRAIN_DATA_FILE $MODEL_OUTPUT_DIR \
                $BERT_EPOCHS $RESULTS_OUTPUT_DIR contexts/wugz-$num-bert.txt \
                bert bert-base-uncased $TOKEN_FILE
        fi
    fi
done < $EXP_CSV