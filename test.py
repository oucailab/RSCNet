from torch.backends import cudnn

from dataset import *
from report import *
from visualization import *


import parameter

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

parameter._init()

def myTest(datasetType, model, dev):
    device = torch.device(dev if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True
    channels = parameter.get_value('channels')
    windowSize = parameter.get_value('windowSize')
    batch_size = parameter.get_value('batch_size')
    num_workers = parameter.get_value('num_workers')
    random_seed = parameter.get_value('random_seed')
    visualization = parameter.get_value('visualization')
    out_features = parameter.get_value('out_features')
    lidar_or_sar_channels = parameter.get_value('lidar_or_sar_channels')
    model_savepath = parameter.get_value('model_savepath')
    report_path = parameter.get_value('report_path')
    image_path = parameter.get_value('image_path')
    
    net = torch.load(model_savepath[datasetType], map_location=device).to(device)
    train_loader, val_loader, test_loader, trntst_loader, all_loader ,hsi_pca_wight = getMyData(datasetType, channels, windowSize, batch_size, num_workers)
    set_random_seed(random_seed)

    getMyReport(datasetType, net, test_loader, report_path[datasetType], device, model)##原本是test_loader
    if visualization:
        if datasetType == 0 or datasetType == 1 or datasetType == 5 or datasetType == 6 or datasetType == 7:
            getMyVisualization(datasetType, net, all_loader, image_path[datasetType], device, model)#原本是all_loader
        else:
            getMyVisualization(datasetType, net, trntst_loader, image_path[datasetType], device, model)#原本是trntst_loader





