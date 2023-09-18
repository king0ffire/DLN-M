########
#自用time代码
########


import os
import sys
import numpy as np
import torch
import argparse
import torch.utils
import torch.backends.cudnn as cudnn
from PIL import Image
from model import DLN_M_baseline_v2,DLN_M_baseline
import torchvision
from lib.dataset import NaEnhance,DatasetFromFolderEval,NaEnhanceTest,TimeTest
import cv2

parser = argparse.ArgumentParser("DLN-M")
parser.add_argument('--data_path', type=str, default='../autodl-tmp/our485/low', help='data location')
#parser.add_argument('--save_path', type=str, default='./LOLTEST/baselinealltest/mit/gen/1000', help='save location')
parser.add_argument('--model', type=str, default='./weights_1000.pt', help='weights location')
parser.add_argument('--syn', type=bool, default=False, help='synthesis na-only images')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')


opt = parser.parse_args()
#os.makedirs(opt.save_path, exist_ok=True)
if opt.syn:
    os.makedirs(os.path.join(opt.save_path,"synthesis"), exist_ok=True)

#TestDataset= DatasetFromFolderEval(opt.data_path,transform=torchvision.transforms.ToTensor())
TestDataset = TimeTest(opt.data_path,0,False,True)
test_queue = torch.utils.data.DataLoader(TestDataset, batch_size=1,pin_memory=True)

def main():
    if not torch.cuda.is_available():
        print('no gpu device available')
        sys.exit(1)

    model = DLN_M_baseline_v2(input_dim=3, dim=64)
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    model.load_state_dict(torch.load(opt.model))
    model.eval()
    with torch.no_grad():
        timestart = cv2.getTickCount()
        for _, (input, image_name) in enumerate(test_queue):
            # input = input.cuda() #(BCHW)
            #image_name = os.path.basename(image_name[0])
            r = model(input)
            # u_name = '%s.png' % (image_name)
            #print('processing {}'.format(image_name))
        timeend = cv2.getTickCount()
        time = (timeend - timestart) / cv2.getTickFrequency()
        print("不带save")
        print(time)
        print(time / len(test_queue))

if __name__ == '__main__':
    main()
