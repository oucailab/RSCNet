import numpy as np
import time
from operator import truediv
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, cohen_kappa_score
import parameter
import warnings
import torch
import matplotlib.pyplot as plt

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

parameter._init()
warnings.filterwarnings("ignore")


def AA_andEachClassAccuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

# 生成报告
def getReport(datasetType, net, test_loader, report_path, class_names, device, model):

    print(f'Test_device:{device}')
    net.eval()
    count = 0
    
    band_selection_counts = None
    
    for hsi_pca ,hsi ,sar, test_labels in test_loader:
        hsi_pca = hsi_pca.to(device)
        hsi = hsi.to(device)
        sar = sar.to(device)
        if model == 'RSCNet':
            outputs, selected_bands = net(hsi.squeeze(1), hsi_pca.squeeze(1), sar)

        
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        if count == 0:
            y_pred = outputs
            y_true = test_labels
            count = 1
        else:
            y_pred = np.concatenate((y_pred, outputs))
            y_true = np.concatenate((y_true, test_labels))



    classification = classification_report(
        y_true, y_pred, target_names=class_names, digits=4)
    confusion = confusion_matrix(y_true, y_pred)
    oa = accuracy_score(y_true, y_pred)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_true, y_pred)
    
    

    classification = str(classification)
    confusion = str(confusion)
    oa = oa * 100
    each_acc = each_acc * 100
    aa = aa * 100
    kappa = kappa * 100
    print('{} Overall accuracy (%)'.format(oa))
    print('{} Average accuracy (%)'.format(aa))
    print('{} Kappa accuracy (%)'.format(kappa))
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    current_time_log = 'Report time: {}\n'.format(current_time)
    with open(report_path, 'a+') as report:
        report.write('{}'.format(parameter.get_taskInfo()))
        report.write('\n')
        report.write('{}'.format(current_time_log))
        report.write('\n')
        report.write('{} Overall accuracy (%)'.format(oa))
        report.write('\n')
        report.write('{} Average accuracy (%)'.format(aa))
        report.write('\n')
        report.write('{} Kappa accuracy (%)'.format(kappa))
        report.write('\n\n')
        report.write('{}'.format(classification))
        report.write('\n')
        report.write('{}'.format(confusion))
        report.write('\n')

# 生成 Houston2013 数据集的报告
def getHouston2013Report(datasetType, net, test_loader, report_path, device, model):
    '''
    net: 训练好的网络
    test_loader: 测试集
    report_path: 报告保存的位置，包含文件名
    '''
    # Houston2013 数据集的类别名
    houston2013_class_names = ['Healthy grass', 'Stressed grass', 'Synthetic grass', 'Trees', 'Soil', 'Water', 'Residential',
                               'Commercial', 'Road', 'Highway', 'Railway', 'Parking lot 1', 'parking lot 2', 'Tennis court', 'Running track']
    print("Houston2013 Start!")
    getReport(datasetType, net, test_loader, report_path, houston2013_class_names, device, model)
    print("Report Success!")

# 生成 Houston2018 数据集的报告
def getHouston2018Report(datasetType, net, test_loader, report_path, device, model):
    # Houston2018 数据集的类别名
    houston2018_class_names = ['Healthy grass', 'Stressed grass', 'Artificial turf', 'Evergreen trees', 'Deciduous trees', 'Bare earth', 'Water', 'Residential buildings', 'Non-residential buildings',
                               'Roads', 'Sidewalks', 'Crosswalks', 'Major thoroughfares', 'Highways', 'Railways', 'Paved parking lots', 'Unpaved parking lots', 'Cars', 'Trains', 'Stadium seats']
    print("Houston2018 Start!")
    getReport(datasetType, net, test_loader, report_path, houston2018_class_names, device, model)
    print("Report Success!")

# 生成 Trento 数据集的报告
def getTrentoReport(datasetType, net, test_loader, report_path, device, model):
    # Trento 数据集的类别名
    trento_class_names = ['Apple trees', 'Buildings', 'Ground', 'Woods', 'Vineyard', 'Roads']
    print("Trento Start!")
    getReport(datasetType, net, test_loader, report_path, trento_class_names, device, model)
    print("Report Success!")

# 生成 Berlin 数据集的报告
def getBerlinReport(datasetType, net, test_loader, report_path, device, model):
    # Berlin 数据集的类别名
    berlin_class_names = ['Forest', 'Residential Area', 'Industrial Area', 'Low Plants', 'Soil', 'Allotment', 'Commercial Area', 'Water']
    print("Berlin Start!")
    getReport(datasetType, net, test_loader, report_path, berlin_class_names, device, model)
    print("Report Success!")

# 生成 Augsburg 数据集的报告
def getAugsburgReport(datasetType, net, test_loader, report_path, device, model):
    # Augsburg 数据集的类别名
    augsburg_class_names = ['Forest', 'Residential Area', 'Industrial Area', 'Low Plants', 'Allotment', 'Commercial Area', 'Water']
    print("Augsburg Start!")
    getReport(datasetType, net, test_loader, report_path, augsburg_class_names, device, model)
    print("Report Success!")

# 生成 YellowRiverEstuary 数据集的报告
def getYellowRiverEstuaryReport(datasetType, net, test_loader, report_path, device, model):
    # YellowRiverEstuary 数据集的类别名
    yellowRiverEstuary_class_names = ['Spartina Alterniflora', 'Pond', 'Woodland', 'Phragmites', 'Typha Orientalis', 'Intertidal Phragmites', 
                                      'Ecological Reservoir', 'Arable Land', 'Lotus Pond', 'Oil Field', 'Salt Field', 'Suaeda Salsa', 'Yellow River', 'Mixed Area 1', 'Mixed Area 2', 'Mixed Area 3', 'Mudflat', 'Sea']
    print("YellowRiverEstuary Start!")
    getReport(datasetType, net, test_loader, report_path, yellowRiverEstuary_class_names, device, model)
    print("Report Success!")

# 生成 LN01 数据集的报告
def getLN01Report(datasetType, net, test_loader, report_path, device, model):
    # LN01 数据集的类别名
    LN01_class_names = ['Reservoir', 'Seawater', 'Sandy soil', 'Broken bridge', 'Barren grass', 'Highway', 
                                      'Railway', 'Bare soil', 'Mountain vegetation', 'Arable land']
    print("LN01 Start!")
    getReport(datasetType, net, test_loader, report_path, LN01_class_names, device, model)
    print("Report Success!")

# 生成 LN02 数据集的报告
def getLN02Report(datasetType, net, test_loader, report_path, device, model):
    # LN02 数据集的类别名
    LN02_class_names = ['Reed water system', 'Phragmites australis', 'Paddy fields', 'Intertidal muds', 'Liao river', 'Construction land', 
                                      'Aquaculture ponds', 'Suaeda salsa', 'Industrial land']
    print("LN02 Start!")
    getReport(datasetType, net, test_loader, report_path, LN02_class_names, device, model)
    print("Report Success!")
def getMyReport(datasetType, net, test_loader, report_path, device, model):
    if(datasetType == 0):
        getHouston2013Report(datasetType, net, test_loader, report_path, device, model)
    elif(datasetType == 1):
        getHouston2018Report(datasetType, net, test_loader, report_path, device, model)
    elif(datasetType == 2):    
        getTrentoReport(datasetType, net, test_loader, report_path, device, model)
    elif(datasetType == 3):    
        getBerlinReport(datasetType, net, test_loader, report_path, device, model)
    elif(datasetType == 4):    
        getAugsburgReport(datasetType, net, test_loader, report_path, device, model)
    elif(datasetType == 5):    
        getYellowRiverEstuaryReport(datasetType, net, test_loader, report_path, device, model)
    elif(datasetType == 6):    
        getLN01Report(datasetType, net, test_loader, report_path, device, model)
    elif(datasetType == 7):    
        getLN02Report(datasetType, net, test_loader, report_path, device, model)       