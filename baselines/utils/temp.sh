dim_arr=(100 1000 2000 5000)
noise_arr=(0.01 0.1 0.2 0.25)

cp ../../../New_Benchmark_Datasets/Original/magic-telescope_overall.txt ../../../New_Benchmark_Datasets/LowDim/

for dim in "${dim_arr[@]}"
do
  for noise in "${noise_arr[@]}"
  do
	echo "$dim","$noise"
	python lowdim_benchmark.py ../../../New_Benchmark_Datasets/Original/magic-telescope_overall.txt $dim $noise ../../../New_Benchmark_Datasets/LowDim
  done
done


