import datetime

hardware = {'Hardswish': 3.0, 'ReLU': 3.0, 'PReLU': 3.0, 
'Conv2d': 0.5, 'AvgPool2d': 0.1, 'BatchNorm2d': 0.05, 'Linear': 0.4, 
'communication': 2.0, 'LayerChoice': 0.0, 'LearnableAlpha': 3.0}

nonlinear_ops = ['ReLU', 'PReLU', 'Hardswish', 'MaxPool']

LOG_DIR = './logs/'

DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
#time of we run the script
TIME_NOW = datetime.now().strftime(DATE_FORMAT)