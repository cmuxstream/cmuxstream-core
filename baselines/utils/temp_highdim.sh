snr_arr=(10 20 50)

for snr in "${snr_arr[@]}"
do
#	python create_outlier_dataset.py ../../../New_Benchmark_Datasets/Original/letter-recognition_overall.txt 0.2 $snr ../../../New_Benchmark_Datasets/HighDim

#	python create_outlier_dataset.py ../../../New_Benchmark_Datasets/Original/madelon_overall.txt 0.2 $snr ../../../New_Benchmark_Datasets/HighDim

	python create_outlier_dataset.py ../../../New_Benchmark_Datasets/Original/gisette_overall.txt 0.2 $snr ../../../New_Benchmark_Datasets/HighDim

#	python create_outlier_dataset.py ../../../New_Benchmark_Datasets/Original/isolet_overall.txt 0.2 $snr ../../../New_Benchmark_Datasets/HighDim

done


