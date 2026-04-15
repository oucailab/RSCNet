from torch.backends import cudnn


from dataset import *

import torch.optim as optim

import time
from sklearn.metrics import accuracy_score
import thop

import parameter

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import csv

from torchinfo import summary
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from thop import profile


from net_RSCNet import RSCNet


parameter._init()



def train(epochs, lr, model, dev, train_loader, test_loader, out_features, model_savepath, log_path, hsi_pca_wight, datasetType):
    device = torch.device(dev if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    cudnn.benchmark = True
    hsi_pca_wight_tensor = torch.from_numpy(hsi_pca_wight).to(device)

    if  model == 'RSCNet':
        pca_channels = parameter.get_value('channels')
        lidar_or_sar_channels = parameter.get_value('lidar_or_sar_channels')[datasetType]
        in_channels = parameter.get_value('data_channels')[datasetType]
        num_classes = parameter.get_value('out_features')[datasetType]
        net = RSCNet(
            hsi_channels=in_channels,
            pca_channels=pca_channels,
            aux_channels=lidar_or_sar_channels,
            num_classes = num_classes,
            embed_dim = in_channels,
            topk_ratio = 0.2,
            num_rscm_layers = 2
        )


    net.to(device)

#-----------------------------------------------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    max_acc = 0
    sum_time = 0
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    current_time_log = 'start time: {}'.format(current_time)
    getLog(log_path, parameter.get_taskInfo())
    getLog(log_path, '-------------------Started Training-------------------')
    getLog(log_path, current_time_log)
   

    for epoch in range(epochs):

        since = time.time()
        net.train()
        for i, (hsi_pca, hsi, sar, tr_labels) in enumerate(train_loader):
            hsi_pca = hsi_pca.to(device)
            sar = sar.to(device)
            hsi = hsi.to(device)
            tr_labels = tr_labels.to(device)
            optimizer.zero_grad()
            if model == 'RSCNet':
                outputs, selected_bands = net(hsi.squeeze(1), hsi_pca.squeeze(1), sar)
                loss = criterion(outputs, tr_labels)

            loss.backward()
            optimizer.step()

        net.eval()
        count = 0
        for hsi_pca ,hsi ,sar, gtlabels in test_loader:
            hsi_pca = hsi_pca.to(device)
            hsi = hsi.to(device)
            sar = sar.to(device)           
            if model == 'RSCNet':
                outputs, selected_bands = net(hsi.squeeze(1), hsi_pca.squeeze(1), sar)
            
            
            outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
            if count == 0:
                y_pred_test =  outputs
                gty = gtlabels
                count = 1
            else:
                y_pred_test = np.concatenate( (y_pred_test, outputs) )
                gty = np.concatenate( (gty, gtlabels) )
        acc1 = accuracy_score(gty, y_pred_test)



        if acc1 > max_acc:
            if model == 'CHNet':
                torch.save(net.state_dict(), model_savepath)
            else:
                torch.save(net, model_savepath)
            max_acc = acc1

        
        time_elapsed = time.time() - since
        sum_time += time_elapsed
        rest_time = (sum_time / (epoch + 1)) * (epochs - epoch - 1)
        currentTime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        log = currentTime + ' [Epoch: %d] [%.0fs, %.0fh %.0fm %.0fs] [current loss: %.4f] acc: %.4f' %(epoch + 1, time_elapsed, (rest_time // 60) // 60, (rest_time // 60) % 60, rest_time % 60, loss.item(), acc1)
        print(log)
        getLog(log_path, log)



    print('max_acc: %.4f' %(max_acc))  
    finish_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    finish_time_log = 'finish time: {} '.format(finish_time)
    mac_acc_log = 'max_acc: {} '.format(max_acc)
    getLog(log_path, mac_acc_log)
    getLog(log_path, finish_time_log)
    getLog(log_path, '-------------------Finished Training-------------------')

def getLog(log_path, str):
    with open(log_path, 'a+') as log:
        log.write('{}'.format(str))
        log.write('\n')

def myTrain(datasetType, model, dev):
    channels = parameter.get_value('channels')
    windowSize = parameter.get_value('windowSize')
    out_features = parameter.get_value('out_features')
    lr = parameter.get_value('lr')
    epoch_nums = parameter.get_value('epoch_nums')
    batch_size = parameter.get_value('batch_size')
    num_workers = parameter.get_value('num_workers')
    random_seed = parameter.get_value('random_seed')
    model_savepath = parameter.get_value('model_savepath')
    log_path = parameter.get_value('log_path')
    train_loader, val_loader, test_loader, trntst_loader, all_loader, hsi_pca_wight = getMyData(datasetType, channels, windowSize, batch_size, num_workers)
    set_random_seed(random_seed)
    print(f'Train_device={dev}')
    train(epoch_nums, lr, model, dev, train_loader, val_loader, out_features[datasetType], model_savepath[datasetType], log_path[datasetType], hsi_pca_wight, datasetType)
