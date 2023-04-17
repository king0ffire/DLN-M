import argparse
import glob
import itertools
import os
import shutil
import sys
import time
from os import listdir
from os.path import join

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
#from sacred import Experiment
#from sacred.utils import apply_backspaces_and_linefeeds
from skimage.metrics._structural_similarity import structural_similarity as compare_ssim
from skimage.metrics.simple_metrics import peak_signal_noise_ratio as compare_psnr
from torch.utils.data import DataLoader

import lib.pytorch_ssim as pytorch_ssim
from lib.data import get_training_set, is_image_file, get_Low_light_training_set, get_training_set_syn
from lib.utils import TVLoss, print_network, SmoothLoss
from model import DLN_M_baseline,DLN
import logging

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=1000, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=50, help='Snapshots')
parser.add_argument('--start_iter', type=int, default=0, help='Starting Epoch')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=0.0001')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')  #windows bug注意
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=2, type=int, help='number of gpu')
parser.add_argument('--patch_size', type=int, default=128, help='Size of cropped LR image')
parser.add_argument('--save_folder', default='models/', help='Location to save checkpoint models')
parser.add_argument('--isdimColor', default=True, help='synthesis at HSV color space')
parser.add_argument('--isaddNoise', default=True, help='synthesis with noise')
parser.add_argument('--save', default='EXP/', help='synthesis with noise')
parser.add_argument('--debug', type=bool, default=True)
opt = parser.parse_args()

#不使用sacred
opt.save = opt.save + '/' + 'Train-{}'.format(time.strftime("%Y%m%d-%H%M%S"))
os.makedirs(opt.save, exist_ok=True)
model_path = opt.save + '/model_epochs/'
os.makedirs(model_path, exist_ok=True)
image_path = opt.save + '/image_epochs/'
os.makedirs(image_path, exist_ok=True)
syn_path = opt.save+'/synthesis/'
os.makedirs(syn_path, exist_ok=True)
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(opt.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
opt.savefolder=model_path
os.makedirs(os.path.join(opt.save, 'scripts'), exist_ok=True)
for script in glob.glob("*.py"):
    dst_file = os.path.join(opt.save, 'scripts', os.path.basename(script))
    shutil.copyfile(script, dst_file)
for script in glob.glob("./lib/*.py"):
    dst_file = os.path.join(opt.save, 'scripts', os.path.basename(script))
    shutil.copyfile(script, dst_file)


def checkpoint(model, epoch, opt, t):
    try:
        os.stat(opt.save_folder)
    except:
        os.mkdir(opt.save_folder)

    model_out_path = model_path + "EXP_{}_{}.pth".format(t, epoch)
    torch.save(model.state_dict(), model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))
    return model_out_path

'''
def log_metrics(_run, logs, iter, end_str=" "):
    str_print = ''
    for key, value in logs.items():
        _run.log_scalar(key, float(value), iter)
        str_print = str_print + "%s: %.4f || " % (key, value)
    print(str_print, end=end_str)'''


def eval(model, epoch, opt, t):
    print("==> Start testing")
    tStart = time.time()
    trans = transforms.ToTensor()
    channel_swap = (1, 2, 0)
    model.eval()
    test_LL_folder = "datasets/LOL/test/low/"
    test_NL_folder = "datasets/LOL/test/high/"
    #test_est_folder = "outputs/eopch_%s_%04d/" % (t, epoch)
    test_est_folder = opt.save + '/image_epochs/'+ str(epoch)
    try:
        os.stat(test_est_folder)
    except:
        os.makedirs(test_est_folder)

    test_LL_list = [join(test_LL_folder, x) for x in sorted(listdir(test_LL_folder)) if is_image_file(x)]
    test_NL_list = [join(test_NL_folder, x) for x in sorted(listdir(test_LL_folder)) if is_image_file(x)]
    est_list = [join(test_est_folder, x) for x in sorted(listdir(test_LL_folder)) if is_image_file(x)]
    '''est_list=[]
    for x in sorted(listdir(test_LL_folder)):
        if is_image_file(x):
            x=join(str(epoch)+"/",x)
            est_list.append(join(test_est_folder, x))'''

    for i in range(test_LL_list.__len__()):
        with torch.no_grad():
            LL = trans(Image.open(test_LL_list[i]).convert('RGB')).unsqueeze(0).cuda()
            prediction = model(LL)
            prediction = prediction.data[0].cpu().numpy().transpose(channel_swap)
            prediction = prediction * 255.0
            prediction = prediction.clip(0, 255)
            Image.fromarray(np.uint8(prediction)).save(est_list[i])
    psnr_score = 0.0
    ssim_score = 0.0
    for i in range(test_NL_list.__len__()):
        gt = cv2.imread(test_NL_list[i])
        est = cv2.imread(est_list[i])
        psnr_val = compare_psnr(gt, est, data_range=255)
        ssim_val = compare_ssim(gt, est, multichannel=True)
        psnr_score = psnr_score + psnr_val
        ssim_score = ssim_score + ssim_val
    psnr_score = psnr_score / (test_NL_list.__len__())
    ssim_score = ssim_score / (test_NL_list.__len__())
    logging.info("eval time: {:.2f} seconds ==> ".format(time.time() - tStart))
    return psnr_score, ssim_score


def main():

    cuda = opt.gpu_mode
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    torch.manual_seed(opt.seed)

    if cuda:
        torch.cuda.manual_seed(opt.seed)
        cudnn.enabled = True
        cudnn.benchmark = True


    t = time.strftime("%Y%m%d-%H%M%S")

    # =============================#
    #   Prepare training data     #
    # =============================#
    # first use the synthesis data (from VOC 2007) to train the model, then use the LOL real data to fine tune
    logging.info('===> Prepare training data')
    #train_set = get_Low_light_training_set(upscale_factor=1, patch_size=opt.patch_size, data_augmentation=True)
    #train_set = get_training_set("datasets/LOL/train", 1, opt.patch_size, True) # uncomment it to do the fine tuning
    train_set = get_training_set_syn("datasets/LOL/train", syn_path, 1, opt.patch_size, True, debug=opt.debug) # uncomment it to do the fine tuning
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize,
                                      pin_memory=True, shuffle=True, drop_last=True)
    # =============================#
    #          Build model        #
    # =============================#
    logging.info('===> Build model')
    lighten = DLN_M_baseline(input_dim=3, dim=64)
    lighten = torch.nn.DataParallel(lighten)
    lighten.load_state_dict(torch.load("./models/EXP_20230101-195550_1000.pth", map_location=lambda storage, loc: storage), strict=True)
    #lighten.load_state_dict(torch.load("./models/DLN_pretrained.pth"))
    logging.info("args = %s", opt)
    #logging.info("loss = ssim_loss + 0.001 * tv_loss + mse_loss")
    logging.info('---------- Networks architecture -------------')
    print_network(lighten)

    logging.info('----------------------------------------------')
    if cuda:
        lighten = lighten.cuda()

    # =============================#
    #         Loss function       #
    # =============================#
    L1_criterion = nn.L1Loss()
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

    # =============================#
    #         Optimizer            #
    # =============================#
    parameters = [lighten.parameters()]
    optimizer = optim.Adam(itertools.chain(*parameters), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)

    # =============================#
    #         Training             #
    # =============================#
    psnr_score, ssim_score = eval(lighten, 0, opt, t) #没看懂干嘛的
    print(psnr_score)
    for epoch in range(opt.start_iter, opt.nEpochs + 1):
        logging.info('===> training epoch %d' % epoch)
        epoch_loss = [0.,0.,0.]
        lighten.train()

        tStart_epoch = time.time()

        for iteration, batch in enumerate(training_data_loader, 1):
            over_Iter = epoch * len(training_data_loader) + iteration
            optimizer.zero_grad()

            LL_t, NL_t = batch[0], batch[1]
            if cuda:
                LL_t = LL_t.cuda()
                NL_t = NL_t.cuda()

            t0 = time.time()

            pred_t = lighten(LL_t)
            #pred_t = NL_t
            ssim_loss = 1 - ssim(pred_t, NL_t)
            tv_loss = TV_loss(pred_t)
            #mse_loss = mse(pred_t, NL_t)
            #l1_loss = L1_criterion(pred_t,LL_t)
            smooth_loss = smooth(NL_t,pred_t)
            loss = ssim_loss + 0.001 * tv_loss + 0.001 * smooth_loss
            loss.backward()
            optimizer.step()
            t1 = time.time()

            epoch_loss[0] += ssim_loss
            epoch_loss[1] += tv_loss
            epoch_loss[2] += smooth_loss

            if iteration % 10 == 0:
                logging.info("Epoch: %d/%d || Iter: %d/%d " % (epoch, opt.nEpochs, iteration, len(training_data_loader)))
                logs = {
                    "loss": loss.item(),
                    "ssim_loss": ssim_loss.item(),
                    "tv_loss": tv_loss.item(),
                    #"smooth_loss": smooth_loss.item()
                    "smooth_loss":smooth_loss.item()
                }
                #log_metrics(_run, logs, over_Iter)
                #print("time: {:.4f} s".format(t1 - t0))
                logging.info("10batch的loss：{} time: {:.4f} s".format(logs,t1-t0))

        logging.info("===> Epoch {} Complete: Avg. Loss: [{:.4f}, {:.4f}, {:.4f}]; ==> {:.2f} seconds".format(epoch, epoch_loss[0] / len(training_data_loader), epoch_loss[1] / len(training_data_loader), epoch_loss[2] / len(training_data_loader), time.time() - tStart_epoch))
        #_run.log_scalar("epoch_loss", float(epoch_loss / len(training_data_loader)), epoch)


        if epoch % (opt.snapshots) == 0:
            file_checkpoint = checkpoint(lighten, epoch, opt, t)
            #exp.add_artifact(file_checkpoint)

            psnr_score, ssim_score = eval(lighten, epoch,opt,t)
            logs = {
                "psnr": psnr_score,
                "ssim": ssim_score,
            }
            #log_metrics(_run, logs, epoch, end_str="\n")
            logging.info("{}epoch:{}".format(opt.snapshots,logs))

        if (epoch + 1) % (opt.nEpochs * 2 / 3) == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 10.0
            logging.info('G: Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))

if __name__ == '__main__':
    main()
