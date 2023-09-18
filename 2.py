########
#自用输出代码
########


import os
import sys
import glob
import cv2
import numpy as np
import torch
import argparse
import torch.utils
import torch.backends.cudnn as cudnn
from PIL import Image
from model import DLN_M_baseline_v2,DLN_M_baseline,DLN_M_Att
import torchvision
from lib.dataset import NaEnhance,DatasetFromFolderEval,NaEnhanceTest
import torchmetrics
import csv

parser = argparse.ArgumentParser("DLN-M")
parser.add_argument('--data_path', type=str, default='../LOLTESTOUT/lollow+8x/', help='data location')
parser.add_argument('--high_path', type=str, default='../LOLTESTOUT/lollow+8x/', help='data location')
parser.add_argument('--save_path', type=str, default='../LOLTESTOUT/DLN/Baseline_v1', help='save location')
parser.add_argument('--model', type=str, default='./models/EXP_20230101-195550_1000.pth', help='weights location')
parser.add_argument('--syn', type=bool, default=True, help='synthesis na-only images')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')


opt = parser.parse_args()
os.makedirs(opt.save_path, exist_ok=True)
if opt.syn:
    os.makedirs(os.path.join(opt.save_path,"synthesis"), exist_ok=True)

#TestDataset= DatasetFromFolderEval(opt.data_path,transform=torchvision.transforms.ToTensor())
TestDataset = NaEnhanceTest(opt.data_path,0,False,True)
test_queue = torch.utils.data.DataLoader(TestDataset, batch_size=1,pin_memory=True)
HighDataset = NaEnhanceTest(opt.high_path,0,False,True)
high_queue = torch.utils.data.DataLoader(TestDataset, batch_size=1,pin_memory=True)

def main():
    if not torch.cuda.is_available():
        print('no gpu device available')
        sys.exit(1)

    model = DLN_M_Att(input_dim=3, dim=64)
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    model.load_state_dict(torch.load(opt.model))
    model.eval()

    file = open(opt.save_path + '/log.csv', mode="w", newline="")
    csvf = csv.writer(file)
    csvf.writerow(["name","","PSNR","","SSIM",""])
    file.flush()
    p=torchmetrics.PeakSignalNoiseRatio()
    s=torchmetrics.StructuralSimilarityIndexMeasure
    with torch.no_grad():
        for _, (input, na, image_name) in enumerate(test_queue):
            #input = input.cuda() #(BCHW)
            (high,n,high_name)=high_queue.__next__()
            image_name = os.path.basename(image_name[0])
            r = model(input)
            #u_name = '%s.png' % (image_name)
            print('processing {}'.format(image_name))
            u_path = os.path.join(opt.save_path,image_name)
            torchvision.utils.save_image(r, u_path)
            psnr=p(r,high)
            ssim=s(r,high)

            if opt.syn:
                path=os.path.join(os.path.join(opt.save_path,"synthesis"),image_name)
                print(na)
                torchvision.utils.save_image(na*input,path)









if __name__ == '__main__':
    main()
