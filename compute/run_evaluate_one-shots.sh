#!/bin/bash
# Runs the performance evaluation procedure for all the fine-tuned models in the `one_shot` directory. 
# This script simply loops through all the fine-tuned models and calls the respective compute script. 
# All printed output is logged to `./outputs/${MODEL_TYPE}/${NUMBER}/${SENT_TYPE}/${TEMPLATE}`, e.g., `./outputs/gpt/sg/sv_agreement/subjrelclause`
# All results for evaluating the models will be sent to `../transformer_evals/fine_tuned`

MODELS_PATH=../one_shot/models
NUMBER=${1}
GPU_NUM=${2}
MODEL_TYPE=${3}
SENT_TYPES=(sv_agreement sv_agreement sv_agreement sv_agreement sv_agreement sv_agreement sv_agreement anaphora anaphora anaphora anaphora)
TEMPLATES=(simple subjrelclause shortvpcoord sentcomp pp objrelclausethat objrelclausenothat simple sentcomp objrelclausethat objrelclausenothat)

if [ $# -ne 3 ]
then
	echo "Needs three arguments, first is 'sg'/'pl', second is gpu num, third is model type 'bert'/'gpt'"
	exit
fi

if [ ${2} -lt 0 ] || [ ${2} -gt 1 ]
then
	echo "GPU number ${2} out of range (needs to be 0 or 1)"
	exit
fi

if [ ${3} != "bert" ] && [ ${3} != "gpt" ]
then
	echo "Model type ${3} invalid (needs to be 'bert' or 'gpt')"
	exit
fi

if [ ! -d ../transformer_evals/fine_tuned]
then
	mkdir ../transformer_evals/fine_tuned
fi

date
echo "Running ${MODEL_TYPE} one-shot models for ${NUMBER}. "
CATEGORIES=($(ls ${MODELS_PATH}/${NUMBER}))
end=${#SENT_TYPES[@]}
for CATEGORY in ${CATEGORIES[@]}
do
	MODELS=($(ls ${MODELS_PATH}/${NUMBER}/${CATEGORY} | grep ${MODEL_TYPE}))
	for MODEL in ${MODELS[@]}
	do
		if [ ! -d ${MODELS_PATH}/../results/${NUMBER}/${CATEGORY}/${MODEL} ] # only run for new models
		then
			i=0
			while [[ $i -lt $end ]]
			do 
				TEMPLATE=${TEMPLATES[$i]}
				SENT_TYPE=${SENT_TYPES[$i]}
				echo "Running python3 -u ${MODEL_TYPE}_compute_sent_probs.py -s ${SENT_TYPE} -t ${TEMPLATE} -g ${GPU_NUM} -m ${MODELS_PATH}/${NUMBER}/${CATEGORY}/${MODEL} > outputs/${MODEL_TYPE}/${NUMBER}/${TEMPLATE}/${SENT_TYPE}/${MODEL}.out"
				mkdir -p outputs/${MODEL_TYPE}/${NUMBER}/${SENT_TYPE}/${TEMPLATE}
				date
				python3 -u ${MODEL_TYPE}_compute_sent_probs.py -s ${SENT_TYPE} -t ${TEMPLATE} -g ${GPU_NUM} -m ${MODELS_PATH}/${NUMBER}/${CATEGORY}/${MODEL} > outputs/${MODEL_TYPE}/${NUMBER}/${TEMPLATE}/${SENT_TYPE}/${MODEL}.out
				echo " "
				date
				tail -n 3 outputs/${MODEL_TYPE}/${NUMBER}/${TEMPLATE}/${SENT_TYPE}/${MODEL}.out
				echo "---------------------------------"
				echo " "
				echo " "
				((i = i + 1))
			done
		fi
	done
done

echo "all done with this bash script :)"
date
