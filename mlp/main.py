import sys, os, time, datetime, argparse
import torch
import torch.nn as nn

import utils
from dataloader import get_dataloader
from train import train, validate

from models.LogisticRegression import LogisticRegression


import pdb

parser = argparse.ArgumentParser(description='Genome Project')
parser.add_argument('--lr', default=2e-3, type=float, metavar='N', help='learning rate, default: 2e-3')
parser.add_argument('--lr_decay_factor',   default=0.99,  type=float, metavar='N', help='lr decay, default: 0.99 (no decay)')
parser.add_argument('--lr_decay_patience', default=10,    type=float, metavar='N', help='lr decay patience, default: 10')
parser.add_argument('--lr_decay_cooldown', default=5,     type=float, metavar='N', help='lr decay cooldown, default: 5')
parser.add_argument('--b', default=16, type=int, metavar='N', help='batch size, default: 16')
parser.add_argument('--dp', default=0.50, type=float, metavar='N', help='dropout probability, default: 0.50')
parser.add_argument('--data', metavar='DIR', default='../data/data_pca_1000comps.pkl', help='path to PCA data')
parser.add_argument('--model', metavar='DIR', default=None, help='path to model, default: None')
parser.add_argument('--epochs', metavar='N', default=300, help='number of epochs, default: 300')
parser.add_argument('--savepath', metavar='DIR', default=None, help='directory to save model and logs')
parser.add_argument('--test_size', metavar='float', default=0.33, help='fraction of training data to use as test')
parser.add_argument('--print_freq', metavar='N', default=100, help='printing/logging frequency')
parser.add_argument('--random_seed', metavar='N', default=42, help='random seed for train/test split')
parser.add_argument('-e', '--eval', dest='evaluate', action='store_true', help='evaluate and do not train, default: False')

def main():
    global opt
    opt = parser.parse_args()
    use_gpu = torch.cuda.is_available()

    # Data
    time_data = time.time()
    train_loader, val_loader, input_size, num_classes = get_dataloader(opt.data, opt.b, opt.test_size, opt.random_seed)

    # Model 
    model = LogisticRegression(input_size, num_classes)
            
    # Pretrained / Initialization
    if opt.model is not None and os.path.isfile(opt.model):
        model.load_state_dict(torch.load(opt.model))
        print('Loaded pretrained model.')
    else:
        for p in model.parameters():
            p.data.uniform_(-0.1, 0.1)
        print('Initialized model from scratch.')
    model = model.cuda() if use_gpu else model

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss(size_average=False)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr) 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=opt.lr_decay_patience, 
                    factor=opt.lr_decay_factor, verbose=True, cooldown=opt.lr_decay_cooldown)

    # Logging
    if opt.savepath == None:
        path = os.path.join('saves', datetime.datetime.now().strftime("%m-%d-%H-%M-%S"))
    else:
        path = opt.savepath
    os.makedirs(path, exist_ok=True)
    logger = utils.Logger(path)
    logger.log('COMMAND LINE ARGS: ' + ' '.join(sys.argv), stdout=False)
    logger.log('ARGS: {}\nOPTIMIZER: {}\nLEARNING RATE: {}\nSCHEDULER: {}\nMODEL: {}\n'.format(
        opt, optimizer, opt.lr, vars(scheduler), model), stdout=False)

    # Train, validate, or predict
    start_time = time.time()
    if opt.evaluate:
        total_correct, total = validate(model, val_loader, criterion)
        logger.log('Accuracy: {:.3f} \t Total correct: {} \t Total: {}'.format(
            total/total_correct, total_correct, total))
    else:
        train(model, train_loader, val_loader, optimizer, criterion, logger, 
                num_epochs=opt.epochs, print_freq=opt.print_freq)
    logger.log('Finished in {}'.format(time.time() - start_time))
    return

if __name__ == '__main__':
    print(' '.join(sys.argv))
    main()

