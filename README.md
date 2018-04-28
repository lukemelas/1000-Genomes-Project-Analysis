# 1000 Genomes Project Analysis

In this repository, we analyze the 1000 genomes project data.

To download the data, run `download.sh`. To run our model, `cd` into the mlp folder. Run `python main.py -h` to see all available options.

```
usage: main.py [-h] [--lr N] [--lr_decay_factor N] [--lr_decay_patience N]
               [--lr_decay_cooldown N] [--b N] [--wd N] [--dp N] [--arch ARCH]
               [--seed N] [--data DIR] [--label DIR] [--model DIR]
               [--epochs N] [--verbose] [--features N] [--savepath DIR]
               [--print_freq N] [--val_fraction float] [--pca_components N]
               [--cross_val_splits N] [-e] [-t]

optional arguments:
  -h, --help            show this help message and exit
  --lr N                learning rate, default=5e-3
  --lr_decay_factor N   lr decay, default=0.99 (no decay)
  --lr_decay_patience N lr decay patience, default=10
  --lr_decay_cooldown N lr decay cooldown, default=5
  --b N                 batch size, default=128
  --wd N                weight decay, default=0
  --dp N                dropout probability, default=0.50
  --arch ARCH           which model to use: MLP|Exp|LogReg, default=MLP
  --seed N              random seed for train/test split, default=-1 (random)
  --data DIR            path to raw (np array) data
  --label DIR           path to raw (np array) labels
  --model DIR           path to model, default=None
  --epochs N            number of epochs, default=600
  --verbose             print more frequently
  --features N          number of features to use, default=-1 (all)
  --savepath DIR        directory to save model and logs
  --print_freq N        printing/logging frequency, default=100
  --val_fraction float  fraction of train to use as val, default=0.2
  --pca_components N    number of components for PCA, default=200
  --cross_val_splits N  number of times to cross-validate, default=5
  -e, --eval            evaluate and do not train, default: False
  -t, --test            evaluate on the test set after training, default: False
  ```
