#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
BERT_SENTIMENT_DIR="/Medical_Sieve_Model_Pipeline/bert_sentiment"

cd $SCRIPT_DIR$BERT_SENTIMENT_DIR
python3 run_bert.py --do_test

cd $SCRIPT_DIR
printf "Sentiment model predicting process has been done!\n"
