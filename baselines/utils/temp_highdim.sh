snr_arr=(1.2 10 20 30 50)

for snr in "${snr_arr[@]}"
do
	python create_outlier_dataset.py ../../../New_Benchmark_Datasets/Original/letter-recognition_overall.txt 0.3 $snr 30.0 ../../../New_Benchmark_Datasets/HighDim/New2/

	python create_outlier_dataset.py ../../../New_Benchmark_Datasets/Original/madelon_overall.txt 0.05 $snr 5.0 ../../../New_Benchmark_Datasets/HighDim/New2/

	python create_outlier_dataset.py ../../../New_Benchmark_Datasets/Original/gisette_overall.txt 0.3 $snr 30.0 ../../../New_Benchmark_Datasets/HighDim/New2/

	python create_outlier_dataset.py ../../../New_Benchmark_Datasets/Original/isolet_overall.txt 0.3 $snr 30.0 ../../../New_Benchmark_Datasets/HighDim/New2/

done


