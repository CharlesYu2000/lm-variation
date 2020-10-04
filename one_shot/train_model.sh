#!/bin/bash

echo 'Training...'
python3 run_language_modeling.py \
	--do_train \
	--line_by_line \
	--overwrite_output_dir \
	--no_cuda \
    ${@:1}
