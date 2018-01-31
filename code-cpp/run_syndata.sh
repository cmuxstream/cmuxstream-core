#make clean
make optimized
K=100
C=100
D=15
N=5000 # scoring interval
I=3082
cat ../data/synDataNoisy.tsv |\
./xstream --fixed --nwindows 0\
  --k $K --c $C --d $D\
  --initsample $I\
  --scoringbatch $N\
  > ../results/scores_syndata_k"$K"_c"$C"_d"$D"_n"$N"_i"$I".txt

python ap_over_time.py ../data/synDataNoisy.tsv\
  ../results/scores_syndata_k"$K"_c"$C"_d"$D"_n"$N"_i"$I".txt
