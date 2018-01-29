dir="/Users/hemanklamba/Documents/Experiments/HighDim_Outliers/New_Benchmark_Datasets/ODDS/New_DS/"
#for ds_name in "${ds_names[@]}"
echo $dir
for ds_name in $dir/*
do
        bs_name=$(basename "$ds_name")
        echo $bs_name
        python RSHash.py "$bs_name"  10
done

