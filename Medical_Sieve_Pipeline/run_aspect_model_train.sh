#!/usr/bin/env bash

SCRIPT_DIR=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
BERT_ASPECT_DIR="/Medical_Sieve_Model_Pipeline/bert_aspect"


cd $SCRIPT_DIR
python3 pooled_rnn_aspect_model.py --do_train
python3 pooled_rnn_text_cnn_aspect_model.py --do_train

cd $SCRIPT_DIR$BERT_ASPECT_DIR
python3 run_bert.py --do_train

cd $SCRIPT_DIR
python3 ensemble_model.py --do_train
printf "Aspect model training process has been done!"
