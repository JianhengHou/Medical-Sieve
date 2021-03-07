#!/usr/bin/env bash

SCRIPT_DIR=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
CRAWLER_DIR="/Medical_Sieve_Model_Pipeline/Medical_Sieve_Model_Pipeline/crawler"

cd $SCRIPT_DIR$CRAWLER_DIR
# To excute the PatientDiscussionSpider
scrapy crawl patient
cd ../../../..
