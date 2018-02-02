#make clean
make optimized
K=100
C=100
D=15
N=20000 # scoring interval
I=20000
cat ../data/url_svmlight/AllDays.svm |\
./xstream --nwindows=1\
  --k $K --c $C --d $D\
  --initsample $I\
  --scoringbatch $N\
  --score-once\
  > ../results/scores_url_scoreonce_window_k"$K"_c"$C"_d"$D"_n"$N"_i"$I".txt

python ap_over_time.py ../data/url_svmlight/AllDaysLabels.svm\
  ../results/scores_url_scoreonce_window_k"$K"_c"$C"_d"$D"_n"$N"_i"$I".txt\
  > ../results/ap_url_scoreonce_window_k"$K"_c"$C"_d"$D"_n"$N"_i"$I".txt

cat ../results/ap_url_scoreonce_window_k"$K"_c"$C"_d"$D"_n"$N"_i"$I".txt
