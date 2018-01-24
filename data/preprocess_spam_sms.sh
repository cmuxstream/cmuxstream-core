#!/usr/bin/env bash

echo "Fetching and processing spam-sms..."
wget "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
unzip smsspamcollection.zip
python preprocess_spam_sms.py
echo "Preprocessed count-vectorized file: spam-sms-preprocessed-counts.tsv.gz"
echo "Preprocessed TFIDF-vectorized file: spam-sms-preprocessed-tfidfs.tsv.gz"
