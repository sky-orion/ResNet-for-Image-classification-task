
import argparse


config = argparse.ArgumentParser()

config.add_argument('--mode', default='train', type=str)

# datasets 
config.add_argument('-dataset_path',type=str,default=r'D:\QQ\1392776996\FileRecv\2021MCM_ProblemC_Files\2021MCM_ProblemC_Files')
config.add_argument('-dataset_txt_path',type=str,default=r'C:\Users\13927\Desktop\数模\model\dataset\lable.txt')
config.add_argument('-train_txt_path',type=str,default=r'C:\Users\13927\Desktop\数模\model\dataset\trainlable.txt')
config.add_argument('-test_txt_path',type=str,default=r'C:\Users\13927\Desktop\数模\model\dataset\unverifiedlable.txt')
config.add_argument('-val_txt_path',type=str,default=r'C:\Users\13927\Desktop\数模\model\dataset\vallable.txt')

# optimizer
config.add_argument('--optimizer',default='adam',choices=['sgd','rmsprop','adam','radam'])
config.add_argument("--lr",type=float,default=0.001)
config.add_argument('--lr-fc-times', '--lft', default=5, type=int,
                    metavar='LR', help='initial model last layer rate')
config.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
config.add_argument('--no_nesterov', dest='nesterov',
                         action='store_false',
                         help='do not use Nesterov momentum')
config.add_argument('--alpha', default=0.99, type=float, metavar='M',
                         help='alpha for ')
config.add_argument('--beta1', default=0.9, type=float, metavar='M',
                         help='beta1 for Adam (default: 0.9)')
config.add_argument('--beta2', default=0.999, type=float, metavar='M',
                         help='beta2 for Adam (default: 0.999)')
config.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
config.add_argument('--gmma', default=2, type=int,help='gmma for FocalLoss')
config.add_argument('--alphaFocalLoss', default=0.5, type=int,help='alpha for FocalLoss')

# training
config.add_argument("--checkpoint",type=str,default='./checkpointsfocal')
config.add_argument("--resume",default=False,type=str,metavar='PATH',help='path to save the latest checkpoint')
config.add_argument("--batch_size",type=int,default=16)
config.add_argument("--start_epoch",default=0,type=int,metavar='N')
config.add_argument('--epochs',default=50,type=int,metavar='N')


config.add_argument('--image-size',type=int,default=512)
config.add_argument('--arch',default='resnet18',choices= ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d'])
config.add_argument('--num_classes',default=2,type=int)

# model path
config.add_argument('--model_path',default=r'C:\Users\13927\Desktop\数模\model\checkpoints\model_37_9288_8333.pth',type=str)
config.add_argument('--result_csv',default=r'C:\Users\13927\Desktop\数模\model\result1.csv')

#device and dataset
config.add_argument('--device', type=int, default='0')
config.add_argument('--useSplit_datatset', type=bool, default=False,help='if use dataset spilt')
config.add_argument('--testratio', type=bool, default=0.2, help='test ratio for whole dataset')
config.add_argument('--valratio', type=bool, default=0.2,  help='val ratio for whole dataset')
config.add_argument('--uesfocalloss', type=bool, default=True,help='if use focalloss ')
config.add_argument('--userocauc', type=bool, default=False,help='if use userocauc ')
args = config.parse_args()