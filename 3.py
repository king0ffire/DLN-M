########
#自用输出代码
########


import os
import sys
import numpy as np
import torch
import argparse
import torch.utils
import torch.backends.cudnn as cudnn
from PIL import Image
from model import DLN_M_baseline_v2,DLN_M_baseline,DLN_M_Att_inter,DLN_M_v2_inter
import torchvision
from lib.dataset import NaEnhance,DatasetFromFolderEval,NaEnhanceTest

parser = argparse.ArgumentParser("DLN-M")
parser.add_argument('--data_path', type=str, default='../LOLTESTOUT/lollow+8x/', help='data location')
parser.add_argument('--save_path', type=str, default='../LOLTESTOUT/DLN/inter', help='save location')
parser.add_argument('--model', type=str, default='./models/weights_1000.pt', help='weights location')
parser.add_argument('--syn', type=bool, default=False, help='synthesis na-only images')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')


opt = parser.parse_args()
os.makedirs(opt.save_path, exist_ok=True)
if opt.syn:
    os.makedirs(os.path.join(opt.save_path,"synthesis"), exist_ok=True)

#TestDataset= DatasetFromFolderEval(opt.data_path,transform=torchvision.transforms.ToTensor())
TestDataset = NaEnhanceTest(opt.data_path,0,False,True)
test_queue = torch.utils.data.DataLoader(TestDataset, batch_size=1,pin_memory=True)

def main():
    if not torch.cuda.is_available():
        print('no gpu device available')
        sys.exit(1)

    model = DLN_M_v2_inter(input_dim=3, dim=64)
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    model.load_state_dict(torch.load(opt.model))
    model.eval()
    with torch.no_grad():
        for _, (input, na, image_name) in enumerate(test_queue):
            #input = input.cuda() #(BCHW)
            image_name = os.path.basename(image_name[0])
            r,Yp,Xp,Xr,Yr = model(input)
            r=torch.clamp(r,0,1)
            Yp=torch.clamp(Yp,0,1)
            Xp=torch.clamp(Xp,0,1)
            Xr=torch.clamp(Xr,0,1)
            Yr=torch.clamp(Yr,0,1)
            #u_name = '%s.png' % (image_name)
            print('processing {}'.format(image_name))
            file=os.path.basename(os.path.splitext(image_name)[0]) #无后缀的名字
            u_path = os.path.join(opt.save_path,image_name)

            torchvision.utils.save_image(r, u_path)
            torchvision.utils.save_image(Yp[0][0], os.path.join(opt.save_path,file+"_Yp.jpg"))
            torchvision.utils.save_image(Xp[0][0], os.path.join(opt.save_path,file+"_Xp.jpg"))
            torchvision.utils.save_image(Xr[0][0], os.path.join(opt.save_path,file+"_Xr.jpg"))
            torchvision.utils.save_image(Yr[0][0], os.path.join(opt.save_path,file+"_Yr.jpg"))
            if opt.syn:
                path=os.path.join(os.path.join(opt.save_path,"synthesis"),image_name)
                print(na)
                torchvision.utils.save_image(na*input,path)



if __name__ == '__main__':
    main()
