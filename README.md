# 1000 Genomes Project Analysis

In this repository, we analyze the 1000 genomes project data.

To download the data, run `download.sh`. To run our model, `cd` into the mlp folder. Run `python main.py -h` to see all available options.

usage: main.py [-h] [--lr N] [--lr_decay_factor N] [--lr_decay_patience N]
               [--lr_decay_cooldown N] [--b N] [--dp N] [--data DIR]
               [--model DIR] [--epochs N] [--savepath DIR] [--val_size float]
               [--test_size float] [--print_freq N] [--cross_val_splits N]
               [-e]

optional arguments:
  -h, --help            show this help message and exit
  --lr N                learning rate, default: 2e-3
  --lr_decay_factor N   lr decay, default: 0.99 (no decay)
  --lr_decay_patience N lr decay patience, default: 10
  --lr_decay_cooldown N lr decay cooldown, default: 5
  --b N                 batch size, default: 16
  --dp N                dropout probability, default: 0.50
  --data DIR            path to PCA data
  --model DIR           path to model, default: None
  --epochs N            number of epochs, default: 300
  --savepath DIR        directory to save model and logs
  --val_size float      fraction of data to use as val
  --test_size float     fraction of data to use as test
  --print_freq N        printing/logging frequency
  --cross_val_splits N  number of times to cross-validate
  -e, --eval            evaluate and do not train, default: False
