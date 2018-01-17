#!/usr/bin/env bash

cat kddcup.data | grep "smtp\|http" | cut -d, -f1,5-6,8-11,13-20,23-\
 | sed "s/normal\./0/g" | sed "s/[[:alpha:]]*\.$/1/g" > http_smtp_continuous.csv
