#make clean
make optimized
K=100
C=100
D=15
N=1000 # scoring interval
I=3082
./xstream --input ../data/synDataNoisy.tsv\
  --k $K --c $C --d $D\
  --fixed\
  --nwindows 0\
  --initsample $I\
  --scoringbatch $N\
  --scoreonce\
  > ../results/scores_syndata_scoreonce_k"$K"_c"$C"_d"$D"_n"$N"_i"$I".txt

python ap_over_time.py ../data/synDataNoisy.tsv\
  ../results/scores_syndata_k"$K"_c"$C"_d"$D"_n"$N"_i"$I".txt
