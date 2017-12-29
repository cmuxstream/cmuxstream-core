cd iForest
python iForest.py magic-telescope_overall.txt_5000.0_0.2_NOISY 10
python iForest.py magic-telescope_overall.txt_5000.0_0.25_NOISY 10
cd ..
cd LODA
python loda_runner.py magic-telescope_overall.txt_5000.0_0.2_NOISY 10
python loda_runner.py magic-telescope_overall.txt_5000.0_0.25_NOISY 10
cd ..
cd RS_Hash
python RSHash.py magic-telescope_overall.txt_5000.0_0.2_NOISY 10
python RSHash.py magic-telescope_overall.txt_5000.0_0.25_NOISY 10
cd ..
cd HSTrees
python HSTree_runner.py magic-telescope_overall.txt_5000.0_0.2_NOISY 10
python HSTree_runner.py magic-telescope_overall.txt_5000.0_0.25_NOISY 10

