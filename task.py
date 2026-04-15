from train import *
from test import *

import parameter

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

parameter._init()


def myTask(lr, epoch_nums, datasetType, dev):
    parameter.set_value('epoch_nums', epoch_nums)
    parameter.set_value('lr', lr)
    parameter.set_value('visualization', visualization)
    myTrain(datasetType, net, dev)
    myTest(datasetType, net, dev)


net = 'RSCNet'    
visualization = True
dev = 'cuda:0'
myTask(0.0001, 1, 3, dev)


