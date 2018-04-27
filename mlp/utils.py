import os
import torch
from torch.autograd import Variable as V

def get_num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Logger():
    '''Prints to a log file and to standard output''' 
    def __init__(self, path):
        if os.path.exists(path):
            self.path = path
        else:
            raise Exception('path does not exist')

    def log(self, info, stdout=True):
        with open(os.path.join(self.path, 'log.log'), 'a') as f:
            print(info, file=f)
            if stdout:
                print(info)

    def save_model(self, model_dict, model_name='model.pkl'):
        torch.save(model_dict, os.path.join(self.path, model_name))

