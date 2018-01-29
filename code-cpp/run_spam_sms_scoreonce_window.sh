#!/usr/bin/env bash

echo "Remember to run make clean before this!"
make optimized
K=100
C=100
D=15
N=100 # scoring interval
I=1394 # 25%
./xstream --input ../data/spam-sms-preprocessed-counts.tsv\
  --k $K --c $C --d $D\
  --fixed\
  --nwindows 1\
  --initsample $I\
  --scoringbatch $N\
  --score-once\
  > ../results/scores_spam_sms_counts_scoreonce_window_k"$K"_c"$C"_d"$D"_n"$N"_i"$I".txt

python ap_over_time.py ../data/spam-sms-preprocessed-counts.tsv\
  ../results/scores_spam_sms_counts_scoreonce_window_k"$K"_c"$C"_d"$D"_n"$N"_i"$I".txt\
  > ../results/ap_spam_sms_counts_scoreonce_window_k"$K"_c"$C"_d"$D"_n"$N"_i"$I".txt
