#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
PREPROCESSOR_DIR="/Medical_Sieve_Model_Pipeline"

printf "Now preprocessing the raw data: splitting paragraphs, regex, generating test corpus file"
cd $SCRIPT_DIR$PREPROCESSOR_DIR
python3 run_dataManager.py
