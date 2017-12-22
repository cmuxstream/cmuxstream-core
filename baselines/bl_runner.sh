input_dir="/home/SHARED/BENCHMARK_HighDim_DATA/Consolidated_Irrel"

filenames=`ls $input_dir/*.csv`
for entry in $filenames
do
  filename=`basename "$entry"`
  python iForest/iForest.py $filename 50
  python LODA/loda_runner.py $filename 50
  python RS_Hash/RSHash.py $filename 50
  python HSTrees/HSTree_runner.py $filename 10 
done