dim_arr=(100 1000 2000 5000)
noise_arr=(0.1)

#cp ../../../New_Benchmark_Datasets/Original/magic-telescope_overall.txt ../../../New_Benchmark_Datasets/LowDim/

for dim in "${dim_arr[@]}"
do
  for noise in "${noise_arr[@]}"
  do
	echo "$dim","$noise"
	python lowdim_benchmark.py ../../../New_Benchmark_Datasets/ODDS/DS/vowels_odds.txt $dim $noise ../../../New_Benchmark_Datasets/ODDS/Noisy_DS/
	
	python lowdim_benchmark.py ../../../New_Benchmark_Datasets/ODDS/DS/glass_odds.txt $dim $noise ../../../New_Benchmark_Datasets/ODDS/Noisy_DS/

	python lowdim_benchmark.py ../../../New_Benchmark_Datasets/ODDS/DS/vertebral_odds.txt $dim $noise ../../../New_Benchmark_Datasets/ODDS/Noisy_DS/

	python lowdim_benchmark.py ../../../New_Benchmark_Datasets/ODDS/DS/wbc_odds.txt $dim $noise ../../../New_Benchmark_Datasets/ODDS/Noisy_DS/
  done
done


