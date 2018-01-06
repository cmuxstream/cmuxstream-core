snr_arr=(15)

for snr in "${snr_arr[@]}"
do
	python create_outlier_dataset.py ../../../New_Benchmark_Datasets/Original/letter-recognition_overall.txt 0.10 $snr 10.0 ../../../New_Benchmark_Datasets/HighDim/New/

	python create_outlier_dataset.py ../../../New_Benchmark_Datasets/Original/madelon_overall.txt 0.10 $snr 10.0 ../../../New_Benchmark_Datasets/HighDim/New/

	python create_outlier_dataset.py ../../../New_Benchmark_Datasets/Original/gisette_overall.txt 0.10 $snr 10.0 ../../../New_Benchmark_Datasets/HighDim/New/

	python create_outlier_dataset.py ../../../New_Benchmark_Datasets/Original/isolet_overall.txt 0.10 $snr 10.0 ../../../New_Benchmark_Datasets/HighDim/New/

done


