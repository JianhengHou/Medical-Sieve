#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
BERT_ASPECT_DIR="/Medical_Sieve_Model_Pipeline/bert_aspect"

printf "Prediction stage with the POOLED RNN model..."
cd $SCRIPT_DIR
python3 pooled_rnn_aspect_model.py --do_predict
printf "Prediction stage with the POOLED RNN TEXT CNN model..."
python3 pooled_rnn_text_cnn_aspect_model.py --do_predict

printf "Prediction stage with the BERT model..."
cd $SCRIPT_DIR$BERT_ASPECT_DIR
python3 run_bert.py --do_test

printf "Prediction stage with the ensemble model..."
cd $SCRIPT_DIR
python3 ensemble_model.py --do_predict
printf "Aspect model predicting process has been done!"
