#!/usr/bin/env python

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

messages = []
labels = []

with open("SMSSpamCollection", "r") as f:
    for line in f:
        fields = line.strip().split("\t")
        ham_or_spam = fields[0]
        message = fields[1]

        if ham_or_spam == "ham":
            label = 0
        else:
            label = 1

        messages.append(message)
        labels.append(label)

count_vectorizer = CountVectorizer(strip_accents='unicode', stop_words='english')
tfidf_vectorizer = TfidfVectorizer(strip_accents='unicode', stop_words='english')

counts = csr_matrix.todense(count_vectorizer.fit_transform(messages))
tfidfs = csr_matrix.todense(tfidf_vectorizer.fit_transform(messages))
print "Data shape:", counts.shape

labels = np.array(labels).reshape(-1,1)
counts = np.concatenate((counts, labels), axis=1)
tfidfs = np.concatenate((tfidfs, labels), axis=1)
np.savetxt("spam-sms-preprocessed-counts.tsv.gz", counts, fmt="%d", delimiter="\t")
np.savetxt("spam-sms-preprocessed-tfidfs.tsv.gz", tfidfs, fmt="%.12f", delimiter="\t")
