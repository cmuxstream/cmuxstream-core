#iinput_dir="/Users/hemanklamba/Documents/Experiments/HighDim_Outliers/New_Benchmark_Datasets/LowDim"

input_dir="/Users/hemanklamba/Documents/Experiments/HighDim_Outliers/New_Benchmark_Datasets/ODDS/Noisy_DS"

filenames=`ls $input_dir/*`
for entry in $filenames
do
  filename=`basename "$entry"`
  #cd iForest
  #python iForest.py $filename 10
  #cd ..
  #cd LODA
  #python loda_runner.py $filename 10
  #cd ..
  cd RS_Hash
  python RSHash.py $filename 10
  cd ..
  #cd HSTrees
  #python HSTree_runner.py $filename 10
  #cd ..
done
