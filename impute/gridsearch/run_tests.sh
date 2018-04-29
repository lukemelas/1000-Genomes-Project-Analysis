# LR
python main.py -t --lr 1e-1 --wd 0.0  --dp 0.5 --id 101
python main.py -t --lr 1e-2 --wd 0.0  --dp 0.5 --id 102
python main.py -t --lr 1e-3 --wd 0.0  --dp 0.5 --epochs 750 --id 103
python main.py -t --lr 5e-4 --wd 0.0  --dp 0.5 --epochs 750 --id 104
python main.py -t --lr 1e-4 --wd 0.0  --dp 0.5 --epochs 750 --id 105
python main.py -t --lr 5e-5 --wd 0.0  --dp 0.5 --epochs 1000 --id 106
python main.py -t --lr 1e-5 --wd 0.0  --dp 0.5 --epochs 1000 --id 107
# WD
python main.py -t --lr 5e-4 --wd 1e-1 --dp 0.5 --epochs 750 --id 108
python main.py -t --lr 5e-4 --wd 1e-2 --dp 0.5 --epochs 750 --id 109
python main.py -t --lr 5e-4 --wd 1e-3 --dp 0.5 --epochs 750 --id 110
python main.py -t --lr 5e-4 --wd 1e-4 --dp 0.5 --epochs 750 --id 111
python main.py -t --lr 5e-4 --wd 1e-5 --dp 0.5 --epochs 750 --id 112
python main.py -t --lr 5e-4 --wd 1e-6 --dp 0.5 --epochs 750 --id 113
# DP
python main.py -t --lr 5e-4 --wd 1e-5 --dp 0.0 --epochs 750 --id 114
python main.py -t --lr 5e-4 --wd 1e-5 --dp 0.1 --epochs 750 --id 115
python main.py -t --lr 5e-4 --wd 1e-5 --dp 0.2 --epochs 750 --id 116
python main.py -t --lr 5e-4 --wd 1e-5 --dp 0.4 --epochs 750 --id 117
python main.py -t --lr 5e-4 --wd 1e-5 --dp 0.5 --epochs 750 --id 118
python main.py -t --lr 5e-4 --wd 1e-5 --dp 0.6 --epochs 750 --id 119
python main.py -t --lr 5e-4 --wd 1e-5 --dp 0.7 --epochs 750 --id 120
# PCA
python main.py -t --lr 5e-4 --wd 1e-5 --dp 0.5 --epochs 750 --no_preloaded_splits --pca_components 10 --id 121
python main.py -t --lr 5e-4 --wd 1e-5 --dp 0.5 --epochs 750 --no_preloaded_splits --pca_components 20 --id 122
python main.py -t --lr 5e-4 --wd 1e-5 --dp 0.5 --epochs 750 --no_preloaded_splits --pca_components 50 --id 123
python main.py -t --lr 5e-4 --wd 1e-5 --dp 0.5 --epochs 750 --no_preloaded_splits --pca_components 100 --id 124
python main.py -t --lr 5e-4 --wd 1e-5 --dp 0.5 --epochs 750 --no_preloaded_splits --pca_components 200 --id 125
python main.py -t --lr 5e-4 --wd 1e-5 --dp 0.5 --epochs 750 --no_preloaded_splits --pca_components 300 --id 126
python main.py -t --lr 5e-4 --wd 1e-5 --dp 0.5 --epochs 750 --no_preloaded_splits --pca_components 500 --id 127
python main.py -t --lr 5e-4 --wd 1e-5 --dp 0.5 --epochs 750 --no_preloaded_splits --pca_components 1000 --id 128


