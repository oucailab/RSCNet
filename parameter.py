import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def _init():  
    global parameter
    parameter = {
        # net
        'channels': 30,
        'windowSize': 11,
        'out_features': [15, 20, 6, 8, 7, 18, 10, 9],
        'depth': [[2, 2, 2], [2, 2, 2], 2],
        'k_size': [10, 1, 20, 10, 60, 10,10,10],
        # train
        'cuda': 'cuda0',
        'lr': 0.0001,
        'epoch_nums': 10,
        'batch_size': 128,  
        'num_workers': 4,
        'random_seed': 6,
        'visualization': False,
        'data_channels':[144,50,63,244,180,285,166,144],    
        'lidar_or_sar_channels':[1,1,1,4,4,3,8,8],
        'model_savepath': ['../model/Houston2013_demo.pth',
                           '../model/Houston2018.pth',
                           '../model/Trento.pth',
                           '../model/Berlin_demo.pth',
                        '../model/Augsburg_demo.pth',
                        '../model/YellowRiverEstuary.pth',
                        '../model/LN01.pth',
                        '../model/LN02',],
        'log_path': ['../log/Houston2013_log_demo.txt',
                     '../log/Houston2018_log.txt',
                     '../log/Trento_log.txt',
                     '../log/Berlin_log_demo.txt',
                     '../log/Augsburg_log_demo.txt',
                     '../log/YellowRiverEstuary_log.txt',
                     '../log/LN01_log.txt',
                     '../log/LN02_log.txt',],
        'report_path': ['../report/Houston2013_report_demo.txt',
                        '../report/Houston2018_report.txt',
                        '../report/Trento_report.txt',
                        '../report/Berlin_report_demo.txt',
                        '../report/Augsburg_report_demo.txt',
                        '../report/YellowRiverEstuary_report.txt',
                        '../report/LN01_report.txt',
                        '../report/LN02_report.txt',],
        'image_path': ['../pic/Houston2013.png',
                       '../pic/Houston2018.png',
                       '../pic/Trento.png',
                       '../pic/Berlin.png',
                       '../pic/Augsburg.png',
                       '../pic/YellowRiverEstuary.png',
                       '../pic/LN01.png',
                       '../pic/LN02.png']

    }


def set_value(key, value):
    parameter[key] = value


def get_value(key):
    try:
        return parameter[key]
    except:
        print('读取' + key + '失败\r\n')


def get_taskInfo():
    return '-----------------------taskInfo----------------------- \n lr:\t{} \n epoch_nums:\t{} \n batch_size:\t{} \n window_size:\t{} \n depth:\t{} \n------------------------------------------------------'.format(
        parameter['lr'], parameter['epoch_nums'], parameter['batch_size'], parameter['windowSize'], parameter['depth'])
