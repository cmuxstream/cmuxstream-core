#make clean
make optimized
K=100
C=100
D=15
N=100000 # scoring interval
I=180173 # 25%
cat ../data/http_smtp_continuous.csv |\
./xstream --fixed --nwindows=1\
  --k $K --c $C --d $D\
  --initsample $I\
  --scoringbatch $N\
  --score-once\
  > ../results/scores_http_smtp_continuous_scoreonce_window_k"$K"_c"$C"_d"$D"_n"$N"_i"$I".txt

python ap_over_time.py ../data/http_smtp_continuous.csv\
  ../results/scores_http_smtp_continuous_scoreonce_window_k"$K"_c"$C"_d"$D"_n"$N"_i"$I".txt\
  > ../results/ap_http_smtp_continuous_scoreonce_window_k"$K"_c"$C"_d"$D"_n"$N"_i"$I".txt
