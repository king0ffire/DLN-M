import argparse
import glob
import logging
import os
import shutil
import sys
from datetime import time

from torch.backends import cudnn

from lib import pytorch_ssim
from model import Finetunemodel, DLN_M_baseline_v2
from lib.dataset import LowLightDatasetFromFolder
import torch
from torch.utils.data import DataLoader
import torchvision
from lib.utils import TVLoss, SmoothLoss

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=1000, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=50, help='Snapshots')
parser.add_argument('--start_iter', type=int, default=0, help='Starting Epoch')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=0.0001')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')  # windows bug注意
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=2, type=int, help='number of gpu')
parser.add_argument('--patch_size', type=int, default=128, help='Size of cropped LR image')
parser.add_argument('--save_folder', default='models/', help='Location to save checkpoint models')
parser.add_argument('--isdimColor', default=True, help='synthesis at HSV color space')
parser.add_argument('--isaddNoise', default=True, help='synthesis with noise')
parser.add_argument('--save', default='EXP/', help='synthesis with noise')
parser.add_argument('--debug', type=bool, default=True)
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

if __name__ == '__main__':

    cuda = opt.gpu_mode
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    torch.manual_seed(opt.seed)

    if cuda:
        torch.cuda.manual_seed(opt.seed)
        cudnn.enabled = True
        cudnn.benchmark = True

    L_dir = "./image/source"
    T_dir = "./image/test"

    # savepath="./image/test"
    # os.makedirs(savepath,exist_ok=True)
    logging.info('================> Prepare training data')
    dataloader = DataLoader(LowLightDatasetFromFolder(L_dir, 256, True,True), opt.batchsize, False, num_workers=opt.threads)  # shuffle注意
    testloader = DataLoader(LowLightDatasetFromFolder(T_dir, 0, False,True), 1, False)
    logging.info('================> Build model')
    SCI = Finetunemodel("./SCIweightpath")
    SCI = torch.nn.DataParallel(SCI)
    for k, v in SCI.named_parameters():
        v.requires_grad = False

    lighten = DLN_M_baseline_v2(input_dim=3, dim=64)
    lighten = torch.nn.DataParallel(lighten)

    if cuda:
        SCI.cuda()
        lighten.cuda()

    L1_criterion = torch.nn.L1Loss()
    TV_loss = TVLoss()
    mse = torch.nn.MSELoss()
    ssim = pytorch_ssim.SSIM()
    smooth = SmoothLoss()
    if cuda:
        gpus_list = range(opt.gpus)
        mse = mse.cuda()
        L1_criterion = L1_criterion.cuda()
        TV_loss = TV_loss.cuda()
        ssim = ssim.cuda(gpus_list[0])
        smooth = smooth.cuda()

    optimizer = torch.optim.Adam(lighten.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)
    SCI.eval()

    testinput, image_name = next(iter(testloader))  # 只用一个图test
    for epoch in range(opt.nEpochs):
        logging.info("================> Training Epoch %d" % epoch)
        epoch_losses = [0., 0., 0., 0, ]
        lighten.train()
        for i, image in enumerate(dataloader):
            optimizer.zero_grad()
            image = image.cuda()
            label = SCI(image)
            pred = lighten(image)

            ssim_loss = 1 - ssim(pred, label)
            tv_loss = TV_loss(pred)
            smooth_loss = smooth(label, pred)

            epoch_losses[0] += ssim_loss
            epoch_losses[1] += tv_loss
            epoch_losses[2] += smooth_loss
            loss = ssim_loss + 0.001 * tv_loss + 0.001 * smooth_loss
            epoch_losses[3] += loss
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                logging.info("One Batch Loss for every 10 batches : SSIM = %.4f ,TV = %.4f ,Smooth = %.4f" % (
                ssim_loss, tv_loss, smooth_loss))

        if epoch % 2 == 0:
            logging.info("Epoch Loss: %.4f" % (epoch_losses[3] / len(dataloader)))
            torch.save(lighten.parameters(), os.path.join(model_path, "weights_%d.pt" % epoch))
            lighten.eval()
            with torch.no_grad():
                pred = lighten(testinput)
                #pred = pred *0.5+0.5
                torchvision.utils.save_image(pred, os.path.join(image_path,"%d.jpg"%epoch))
