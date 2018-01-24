#!/usr/bin/env bash

python plot_ap_over_time.py \
  ../results/ap_http_smtp_continuous_k100_c100_d10_n100000_i144138.txt \
  ../results/ap_http_smtp_continuous_k100_c100_d10_n100000_i180173.txt \
  ../results/ap_http_smtp_continuous_k100_c100_d15_n100000_i144138.txt \
  ../results/ap_http_smtp_continuous_k100_c100_d15_n100000_i180173.txt

plot_ap_over_time.py \
  ../results/ap_spam_sms_counts_continuous_k100_c100_d10_n100_i1394.txt \
  ../results/ap_spam_sms_counts_continuous_k100_c100_d15_n100_i1394.txt
