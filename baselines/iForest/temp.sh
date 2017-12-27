#ds_names=('abalone' 'blood-transfusion' 'breast-cancer-wisconsin' 'breast-tissue' 'cardiotocography' 'ecoli' 'gisette' 'glass' 'haberman' 'ionosphere' 'iris' 'isolet' 'letter-recognition' 'libras' 'madelon' 'magic-telescope' 'miniboone' 'musk-2' 'page-blocks' 'parkinsons' 'pendigits' 'pima-indians' 'sonar' 'spect-heart' 'statlog-satimage' 'statlog-segment' 'statlog-shuttle' 'statlog-vehicle' 'synthetic-control-chart' 'vertebral-column' 'wall-following-robot' 'waveform-1' 'waveform-2' 'wine' 'yeast')

dir="/Users/hemanklamba/Documents/Experiments/HighDim_Outliers/New_Benchmark_Datasets/HighDim"
#for ds_name in "${ds_names[@]}"
echo $dir
for ds_name in $dir/*
do 
	bs_name=$(basename "$ds_name")
	echo $bs_name
	python iForest.py "$bs_name"  10
done
