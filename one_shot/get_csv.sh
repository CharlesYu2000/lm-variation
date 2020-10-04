#!/bin/bash

# Get the latest transformer-learning-experiments.csv
wget -O transformer_learning_experiments.csv \
    "https://docs.google.com/spreadsheets/d/1frPBS2ONveShIvYuypV-KmOHiPJJrwksZ02KAlTHyl8/export?gid=0&format=csv"

echo -e "" >> transformer_learning_experiments.csv