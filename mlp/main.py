import sys, os, time, datetime, argparse
import random

import torch
import torch.nn as nn

import utils
from dataloader import get_data, get_dataloader
from train import train, validate

from models.LogisticRegression import LogisticRegression
from models.MLP import MLP
from models.Experimental import ExperimentalModel

from sklearn.model_selection import StratifiedKFold

import pdb

parser = argparse.ArgumentParser(description='Genome Project')
parser.add_argument('--lr', default=5e-3, type=float, metavar='N', help='learning rate, default=5e-3')
parser.add_argument('--lr_decay_factor',   type=float, default=0.99, metavar='N', help='lr decay, default=0.99 (no decay)')
parser.add_argument('--lr_decay_patience', type=int,   default=10,   metavar='N', help='lr decay patience, default=10')
parser.add_argument('--lr_decay_cooldown', type=int,   default=5,    metavar='N', help='lr decay cooldown, default=5')
parser.add_argument('--b', default=128, type=int, metavar='N', help='batch size, default=128')
parser.add_argument('--wd', default=0, type=float, metavar='N', help='weight decay, default=0')
parser.add_argument('--dp', default=0.50, type=float, metavar='N', help='dropout probability, default=0.50')
parser.add_argument('--arch', default='MLP', help='which model to use: MLP|Exp|LogReg, default=MLP')
parser.add_argument('--seed', metavar='N', default=-1, type=int, help='random seed for train/test split, default=-1 (random)')
parser.add_argument('--data', metavar='DIR', default='../data/data_all_float16.pkl', help='path to raw (np array) data')
parser.add_argument('--label', metavar='DIR', default='../data/pops_with_ints_pandas.pkl', help='path to raw (np array) labels')
parser.add_argument('--model', metavar='DIR', default=None, help='path to model, default=None')
parser.add_argument('--epochs', metavar='N', type=int, default=600, help='number of epochs, default=600')
parser.add_argument('--verbose', action='store_true', help='print more frequently')
parser.add_argument('--features', metavar='N', type=int, default=-1, help='number of features to use, default=-1 (all)')
parser.add_argument('--savepath', metavar='DIR', default=None, help='directory to save model and logs')
parser.add_argument('--print_freq', metavar='N', type=int, default=100, help='printing/logging frequency, default=100')
parser.add_argument('--val_fraction', metavar='float', default=0.1, help='fraction of train to use as val, default=0.2')
parser.add_argument('--pca_components', metavar='N', type=int, default=200, help='number of components for PCA, default=200')
parser.add_argument('--cross_val_splits', metavar='N', type=int, default=5, help='number of times to cross-validate, default=5')
parser.add_argument('--no_preloaded_splits', action='store_false', help='do not use preloaded train/val/test splits')
parser.add_argument('-e', '--eval', dest='evaluate', action='store_true', help='evaluate and do not train, default: False')
parser.add_argument('-t', '--test', dest='test', action='store_true', help='evaluate on the test set after training, default: False')

def main():
    global opt
    opt = parser.parse_args()
    use_gpu = torch.cuda.is_available()

    # Set up logging
    if opt.savepath == None:
        path = os.path.join('save', datetime.datetime.now().strftime("%d-%H-%M-%S"))
    else:
        path = opt.savepath
    os.makedirs(path, exist_ok=True)
    logger = utils.Logger(path)

    # Keep track of accuracies 
    val_accuracies = []
    test_accuracies = []

    # Seed for cross-val split
    seed = random.randint(0,10000)
    logger.log('SEED: {}'.format(seed), stdout=False)

    # Load data
    start = time.time()
    data, label = get_data(opt.data, opt.label)
    logger.log('Data loaded in {:.1f}s\n'.format(time.time() - start))

    # Create cross-validation splits
    kf = StratifiedKFold(n_splits=opt.cross_val_splits, random_state=seed)

    # Cross validate 
    for i, (train_index, test_index) in enumerate(kf.split(data, label)):

        # Log split
        logger.log('------------- SPLIT {} --------------\n'.format(i+1))

        # Train / test split 
        data_X, data_X_test = data[train_index], data[test_index]
        y, y_test = label[train_index], label[test_index]

        # Perform PCA and get dataloader
        start = time.time()
        train_loader, val_loader, test_loader, input_size, num_classes = get_dataloader(data_X, 
                data_X_test, y, y_test, opt.b, opt.val_fraction, opt.pca_components, i=i)
        logger.log('Dataloader loaded in {:.1f}s\n'.format(time.time() - start))

        # Model 
        arch = opt.arch.lower()
        assert arch in ['logreg', 'mlp', 'exp']
        if arch == 'logreg': 
            model = LogisticRegression(input_size, num_classes)
        elif arch == 'mlp':
            model = MLP(input_size, num_classes, opt.dp) 
        elif arch == 'exp': 
            model = ExperimentalModel(input_size, num_classes, opt.dp)
        print(model)

        # Pretrained / Initialization
        if opt.model is not None and os.path.isfile(opt.model):
            model.load_state_dict(torch.load(opt.model))
            logger.log('Loaded pretrained model.', stdout=(i==0))
        else:
            for p in model.parameters():
                p.data.uniform_(-0.1, 0.1)
            logger.log('Initialized model from scratch.', stdout=(i==0))
        model = model.cuda() if use_gpu else model

        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss(size_average=False)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr, weight_decay=opt.wd) 
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=opt.lr_decay_patience, 
                        factor=opt.lr_decay_factor, verbose=True, cooldown=opt.lr_decay_cooldown)

        # Log parameters
        logger.log('COMMAND LINE ARGS: ' + ' '.join(sys.argv), stdout=False)
        logger.log('ARGS: {}\nOPTIMIZER: {}\nLEARNING RATE: {}\nSCHEDULER: {}\nMODEL: {}\n'.format(
            opt, optimizer, opt.lr, vars(scheduler), model), stdout=False)

        # Either evaluate model
        if opt.evaluate:
            assert opt.model != None, 'no pretrained model to evaluate'
            total_correct, total, _ = validate(model, val_loader, criterion)
            logger.log('Accuracy: {:.3f} \t Total correct: {} \t Total: {}'.format(
                total_correct/total, total_correct, total))
            return 
        # Or train model
        else:
            start_time = time.time()
            best_acc = train(model, train_loader, val_loader, optimizer, criterion, logger, 
                num_epochs=opt.epochs, print_freq=opt.print_freq, model_id=i)
            logger.log('Best train accuracy: {:.2f}% \t Finished split {} in {:.2f}s\n'.format(
                100 * best_acc, i+1, time.time() - start_time))
            val_accuracies.append(best_acc)

        # Optionally also test on test set
        if opt.test:
            best_model_path = os.path.join(path, 'model_{}.pth'.format(i))
            model.load_state_dict(torch.load(best_model_path)) # load best model
            total_correct, total, _ = validate(model, val_loader, criterion) # check val set
            logger.log('Val Accuracy: {:.3f} \t Total correct: {} \t Total: {}'.format(
                total_correct/total, total_correct, total))
            total_correct, total, visualize = validate(model, test_loader, criterion, visualize=True) # run test set
            logger.log('Test Accuracy: {:.3f} \t Total correct: {} \t Total: {}\n'.format(
                total_correct/total, total_correct, total))
            logger.save_model(visualize, 'visualize_{}.pth'.format(i))
            test_accuracies.append(total_correct/total)
    
    # Log after training
    logger.log('Val Accuracies: {}'.format(val_accuracies))
    logger.log('Test Accuracies: {}'.format(test_accuracies))

if __name__ == '__main__':
    print(' '.join(sys.argv))
    print()
    main()

