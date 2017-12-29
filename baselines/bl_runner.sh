input_dir="/Users/hemanklamba/Documents/Experiments/HighDim_Outliers/New_Benchmark_Datasets/LowDim"

filenames=`ls $input_dir/*`
for entry in $filenames
do
  filename=`basename "$entry"`
  cd iForest
  python iForest.py $filename 1
  cd ..
  cd LODA
  python loda_runner.py $filename 1
  cd ..
  cd RS_Hash
  python RSHash.py $filename 1
  cd ..
  cd HSTrees
  python HSTree_runner.py $filename 1 
  cd ..
done
