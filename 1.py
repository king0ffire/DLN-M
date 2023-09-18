######
#自用训练代码
######
import argparse
import glob
import logging
import os
import shutil
import sys
import time
import csv
from torch.backends import cudnn

from lib import pytorch_ssim
from model import Finetunemodel, DLN_M_baseline_v2,DLN_M_Att,DLN_M_Att_Ab1,DLN_M_Att_Ab2,  DLN_M_Att_Ab3
from lib.dataset import LowLightDatasetFromFolder, NaEnhance
import torch
from torch.utils.data import DataLoader
import torchvision
from lib.utils import TVLoss, SmoothLoss

parser = argparse.ArgumentParser()
parser.add_argument('--batchsize', type=int, default=32, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=1000, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=50, help='Snapshots')
parser.add_argument('--start_iter', type=int, default=0, help='Starting Epoch')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=0.0001')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')  # windows bug注意
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=0, type=int, help='number of gpu')
parser.add_argument('--patchsize', type=int, default=128, help='Size of cropped LR image')
parser.add_argument('--save_folder', default='models/', help='Location to save checkpoint models')
parser.add_argument('--isdimColor', default=True, help='synthesis at HSV color space')
parser.add_argument('--isaddNoise', default=True, help='synthesis with noise')
parser.add_argument('--save', default='EXP/', help='Default save folder')
parser.add_argument('--debug', type=bool, default=True)
parser.add_argument('--lowset', type=str, default='./image/source')
parser.add_argument('--testset', type=str, default='./image/source')
opt = parser.parse_args()

# 不使用sacred
opt.save = opt.save + '/' + 'Train-{}'.format(time.strftime("%Y%m%d-%H%M%S"))
os.makedirs(opt.save, exist_ok=True)
model_path = opt.save + '/model_epochs/'
os.makedirs(model_path, exist_ok=True)
image_path = opt.save + '/image_epochs/'
os.makedirs(image_path, exist_ok=True)
syn_path = opt.save + '/synthesis/'
os.makedirs(syn_path, exist_ok=True)
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(opt.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
opt.savefolder = model_path
os.makedirs(os.path.join(opt.save, 'scripts'), exist_ok=True)
for script in glob.glob("*.py"):
    dst_file = os.path.join(opt.save, 'scripts', os.path.basename(script))
    shutil.copyfile(script, dst_file)
for script in glob.glob("./lib/*.py"):
    dst_file = os.path.join(opt.save, 'scripts', os.path.basename(script))
    shutil.copyfile(script, dst_file)
for script in glob.glob("./lib/pytorch_ssim/*.py"):
    dst_file = os.path.join(opt.save, 'scripts', os.path.basename(script))
    shutil.copyfile(script, dst_file)

if __name__ == '__main__':
    logging.info("running 1.py. 将要训练pyramid att3分支网络，loss：1 0.001 0.001，单Math辅助,SSIM->l1")
    logging.info("Parameters = %s", opt)
    cuda = opt.gpu_mode
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    torch.manual_seed(opt.seed)

    if cuda:
        torch.cuda.manual_seed(opt.seed)
        cudnn.enabled = True
        cudnn.benchmark = True

    # savepath="./image/test"
    # os.makedirs(savepath,exist_ok=True)
    logging.info('================> Prepare training data')
    dataloader = DataLoader(NaEnhance(opt.lowset, opt.patchsize, True, True), opt.batchsize, False,
                            num_workers=opt.threads)  # shuffle注意
    testloader = DataLoader(LowLightDatasetFromFolder(opt.testset, 0, False, True), 1, False)
    logging.info('================> Build model')
    SCI = Finetunemodel("./models/difficult.pt")
    SCI = torch.nn.DataParallel(SCI)
    for k, v in SCI.named_parameters():
        v.requires_grad = False

    lighten = DLN_M_Att(input_dim=3, dim=64)
    lighten = torch.nn.DataParallel(lighten)
    #lighten.load_state_dict(torch.load("./EXP/Train-20230519-213122/model_epochs/weights_1000.pt"))
    pytorch_total_params = sum(p.numel() for p in lighten.parameters())
    logging.info("Number of parameters %d"%pytorch_total_params)

    if cuda:
        SCI.cuda()
        lighten.cuda()

    L1_criterion = torch.nn.L1Loss()
    TV_loss = TVLoss()
    mse = torch.nn.MSELoss()
    ssim = pytorch_ssim.SSIM()
    smooth = SmoothLoss()
    if cuda:
        mse = mse.cuda()
        L1_criterion = L1_criterion.cuda()
        TV_loss = TV_loss.cuda()
        ssim = ssim.cuda()
        smooth = smooth.cuda()

    optimizer = torch.optim.Adam(lighten.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)
    SCI.eval()

    file = open(opt.save + '/log.csv', mode="w", newline="")
    csvf = csv.writer(file)
    csvf.writerow(["epoch", "", "loss1", "","loss2","","loss3","","loss4","","loss_all"])
    file.flush()

    testinput = next(iter(testloader))  # 只用一个图test
    for epoch in range(opt.nEpochs+1):
        logging.info("================> Training Epoch %d" % epoch)
        epoch_losses = [0., 0., 0., 0.,0. ]
        lighten.train()
        for i, (image, na) in enumerate(dataloader):
            optimizer.zero_grad()
            image = image.cuda()

            na = na.cuda()
            #na = torch.sqrt(na)
            na = na.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(image.size()[0],image.size()[1],image.size()[2],image.size()[3])
            label=image*na
            label=torch.clamp(label,0.0,1.0)

            #_, label = SCI(image)

            if opt.debug and epoch==0:
                torchvision.utils.save_image(label, os.path.join(syn_path, "%d_%d.jpg" % (epoch,i)))
                logging.info("debug %d.jpg saved"%i)
                logging.info("na="+str(na[:,0,0,0].tolist()))

            pred = lighten(image)

            ssim_loss = 1-ssim(pred, label)
            #l1_loss=L1_criterion(pred,label)
            tv_loss = TV_loss(pred)
            smooth_loss = smooth(label, pred)

            epoch_losses[0] += ssim_loss
            epoch_losses[1] += tv_loss
            epoch_losses[2] += smooth_loss
            #epoch_losses[3] += l1_loss
            loss = ssim_loss + 0.001 * tv_loss + 0.001 * smooth_loss #+0.01*l1_loss
            epoch_losses[4] += loss
            loss.backward()
            optimizer.step()


            if i % 10 == 0:
                logging.info("One Batch Loss for every 10 batches : loss1 = %.4f ,loss2 = %.4f ,loss3 = %.4f" % (
                    ssim_loss, tv_loss, smooth_loss))

        for i in range(len(epoch_losses)):
            epoch_losses[i]=epoch_losses[i]/len(dataloader)
        logging.info("Epoch average Loss: %.4f" % epoch_losses[4].item())
        csvf.writerow([epoch, "", epoch_losses[0].item(), "", epoch_losses[1].item(), "", epoch_losses[2].item(),"","","",epoch_losses[4].item()])


        if epoch% opt.snapshots == 0:#save模型和图片位置占得多
            file.flush()#把csvf刷新出去
            torch.save(lighten.state_dict(), os.path.join(model_path, "weights_%d.pt" % epoch))
            lighten.eval()
            with torch.no_grad():
                pred = lighten(testinput)
                # pred = pred *0.5+0.5
                torchvision.utils.save_image(pred, os.path.join(image_path, "%d.jpg" % epoch))

        if epoch == opt.nEpochs /2: #dong tai xue xi lv
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 2.0
            logging.info('G: Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))

        if epoch == opt.nEpochs *3/4: #dong tai xue xi lv
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 2.0
            logging.info('G: Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))