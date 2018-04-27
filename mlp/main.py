import sys, os, time, datetime, argparse
import random

import torch
import torch.nn as nn

import utils
from dataloader import get_dataloader
from train import train, validate

from models.LogisticRegression import LogisticRegression
from models.MLP import MLP

import pdb

parser = argparse.ArgumentParser(description='Genome Project')
parser.add_argument('--lr', default=2e-3, type=float, metavar='N', help='learning rate, default: 2e-3')
parser.add_argument('--lr_decay_factor',   type=float, default=0.99, metavar='N', help='lr decay, default: 0.99 (no decay)')
parser.add_argument('--lr_decay_patience', type=int,   default=10,   metavar='N', help='lr decay patience, default: 10')
parser.add_argument('--lr_decay_cooldown', type=int,   default=5,    metavar='N', help='lr decay cooldown, default: 5')
parser.add_argument('--b', default=16, type=int, metavar='N', help='batch size, default: 16')
parser.add_argument('--dp', default=0.50, type=float, metavar='N', help='dropout probability, default: 0.50')
parser.add_argument('--data', metavar='DIR', default='../data/data_pca_1000comps.pkl', help='path to PCA data')
parser.add_argument('--model', metavar='DIR', default=None, help='path to model, default: None')
parser.add_argument('--epochs', metavar='N', type=int, default=300, help='number of epochs, default: 300')
parser.add_argument('--savepath', metavar='DIR', default=None, help='directory to save model and logs')
parser.add_argument('--test_size', metavar='float', default=0.33, help='fraction of training data to use as test')
parser.add_argument('--print_freq', metavar='N', type=int, default=100, help='printing/logging frequency')
parser.add_argument('--random_seed', metavar='N', type=int, default=42, help='random seed for train/test split')
parser.add_argument('--cross_val_splits', metavar='N', type=int, default=1, help='number of times to cross-validate')
parser.add_argument('-e', '--eval', dest='evaluate', action='store_true', help='evaluate and do not train, default: False')

def main():
    global opt
    opt = parser.parse_args()
    use_gpu = torch.cuda.is_available()

    # Logging
    if opt.savepath == None:
        path = os.path.join('save', datetime.datetime.now().strftime("%m-%d-%H-%M-%S"))
    else:
        path = opt.savepath
    os.makedirs(path, exist_ok=True)
    logger = utils.Logger(path)


    # Cross validate 
    seeds = []
    accuracies = []
    for i in range(opt.cross_val_splits):

        # Log split
        logger.log('------------- SPLIT {} --------------\n'.format(i+1))

        # Generate random seed for train/test split
        random_seed = 4 #random.randint(0,1000) if opt.cross_val_splits > 1 else opt.random_seed 
        seeds.append(random_seed) 

        # Data
        time_data = time.time()
        train_loader, val_loader, input_size, num_classes = get_dataloader(opt.data, opt.b, opt.test_size, random_seed)

        # Model 
        model = MLP(input_size, num_classes, opt.dp) # LogisticRegression(input_size, num_classes)
                
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
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr) 
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=opt.lr_decay_patience, 
                        factor=opt.lr_decay_factor, verbose=True, cooldown=opt.lr_decay_cooldown)

        # Log parameters
        logger.log('COMMAND LINE ARGS: ' + ' '.join(sys.argv), stdout=False)
        logger.log('ARGS: {}\nOPTIMIZER: {}\nLEARNING RATE: {}\nSCHEDULER: {}\nMODEL: {}\n'.format(
            opt, optimizer, opt.lr, vars(scheduler), model), stdout=False)

        # Either evaluate model
        if opt.evaluate:
            assert opt.model != None, 'no pretrained model to evaluate'
            total_correct, total = validate(model, val_loader, criterion)
            logger.log('Accuracy: {:.3f} \t Total correct: {} \t Total: {}'.format(
                total/total_correct, total_correct, total))
            return 
        # Or train model
        else:
            start_time = time.time()
            best_acc = train(model, train_loader, val_loader, optimizer, criterion, logger, 
                num_epochs=opt.epochs, print_freq=opt.print_freq, model_id=i)
            logger.log('Best accuracy: {:.2f}% \t Finished split {} in {:.2f}s\n'.format(
                100 * best_acc, i+1, time.time() - start_time))
            accuracies.append(acc)
    
    # Log after training
    logger.log('Seeds: {}'.format(seeds), stdout=False)
    logger.log('Accuracies: {}'.format(', '.join(accuracies)))

if __name__ == '__main__':
    print(' '.join(sys.argv))
    print()
    main()

