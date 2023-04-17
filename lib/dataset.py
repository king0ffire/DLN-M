import os
import random
import sys
from os import listdir
from os.path import join, basename

import cv2
import numpy as np
import torch
import torch.utils.data as data
import torchvision
from PIL import Image, ImageOps, ImageEnhance
from lib.utils import get_na

path = os.getcwd()
sys.path.append(path)



def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".bmp", ".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    # y, _, _ = img.split()
    return img


def rescale_img(img_in, scale):
    size_in = img_in.size
    new_size_in = tuple([int(x * scale) for x in size_in])
    img_in = img_in.resize(new_size_in, resample=Image.BICUBIC)
    return img_in


def get_patch(img_in, img_tar, patch_size, scale, ix=-1, iy=-1): #PIL.Image
    (ih, iw) = img_in.size
    # (th, tw) = (scale * ih, scale * iw)

    patch_mult = scale  # if len(scale) > 1 else 1
    tp = patch_mult * patch_size
    ip = tp // scale

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    (tx, ty) = (scale * ix, scale * iy)

    img_in = img_in.crop((ty, tx, ty + tp, tx + tp))
    img_tar = img_tar.crop((ty, tx, ty + tp, tx + tp))
    # img_bic = img_bic.crop((ty, tx, ty + tp, tx + tp))

    # info_patch = {
    #    'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}

    return img_in, img_tar


def augment(img_in, img_tar, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}

    if random.random() < 0.5 and flip_h:
        img_in = ImageOps.flip(img_in)
        img_tar = ImageOps.flip(img_tar)
        # img_bic = ImageOps.flip(img_bic)
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            img_in = ImageOps.mirror(img_in)
            img_tar = ImageOps.mirror(img_tar)
            # img_bic = ImageOps.mirror(img_bic)
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            img_in = img_in.rotate(180)
            img_tar = img_tar.rotate(180)
            # img_bic = img_bic.rotate(180)
            info_aug['trans'] = True

    return img_in, img_tar, info_aug


class DatasetFromFolder(data.Dataset): #get时才把图片读入
    def __init__(self, HR_dir, LR_dir, patch_size, upscale_factor, data_augmentation,
                 transform=None):
        super(DatasetFromFolder, self).__init__()
        #self.hr_image_filenames = [join(HR_dir, x) for x in listdir(HR_dir) if is_image_file(x)]
        #self.lr_image_filenames = [join(LR_dir, x) for x in listdir(HR_dir) if is_image_file(x)]
        self.lr_image_filenames = [join(LR_dir, x) for x in listdir(LR_dir) if is_image_file(x)]

        self.patch_size = patch_size
        self.upscale_factor = upscale_factor
        self.transform = transform
        self.data_augmentation = data_augmentation

    def __getitem__(self, index):

        target = load_img(self.hr_image_filenames[index])
        # name = self.hr_image_filenames[index]
        # lr_name = name[:25]+'LR/'+name[28:-4]+'x4.png'
        # lr_name = name[:18] + 'LR_4x/' + name[21:]
        input = load_img(self.lr_image_filenames[index])

        # target = ImageOps.equalize(target)
        # input_eq = ImageOps.equalize(input)
        # target = ImageOps.equalize(target)
        #         # input = ImageOps.equalize(input)
        input, target, = get_patch(input, target, self.patch_size, self.upscale_factor)

        if self.data_augmentation:
            img_in, img_tar, _ = augment(input, target)

        if self.transform:
            img_in = self.transform(img_in)
            img_tar = self.transform(img_tar)

        return img_in, img_tar

    def __len__(self):
        return len(self.hr_image_filenames)
        #return len(self.lr_image_filenames)


class DatasetFromFolder_nonsup(data.Dataset): #通过get_na直接增强了图像
    def __init__(self, synstore, LR_dir, patch_size, upscale_factor, data_augmentation, transform=None,debug=False):
        super(DatasetFromFolder_nonsup, self).__init__()
        # self.hr_image_filenames = [join(HR_dir, x) for x in listdir(HR_dir) if is_image_file(x)]
        # self.lr_image_filenames = [join(LR_dir, x) for x in listdir(HR_dir) if is_image_file(x)]
        self.lr_image_filenames = [join(LR_dir, x) for x in listdir(LR_dir) if is_image_file(x)]

        self.patch_size = patch_size
        self.upscale_factor = upscale_factor
        self.transform = transform
        self.data_augmentation = data_augmentation

        self.low=[]
        self.high=[]
        for idx in range(len(self.lr_image_filenames)):
            input = load_img(self.lr_image_filenames[idx])  # type:PIL.Image
            target = np.asarray(input)/255
            natar=target.transpose([2,0,1])
            ratio=0.0
            for i in range(3):#RGB
                ratio+=get_na(natar[i])
            ratio=ratio/3
            target = target*ratio*255
            target=np.clip(target,0,255)
            target = Image.fromarray(target.astype('uint8')).convert("RGB")
            if debug:
                target.save(join(synstore, basename(self.lr_image_filenames[idx])))  #备用查验
                print("Synthtised {} stored, ratio:{:.2f}".format(basename(self.lr_image_filenames[idx]),ratio))
            self.low.append(input)
            self.high.append(target)
            if idx % 50==0:
                print("loading:{}/{}".format(idx,len(self.lr_image_filenames)))

    def __getitem__(self, index):

        #target = load_img(self.hr_image_filenames[index])
        # name = self.hr_image_filenames[index]
        # lr_name = name[:25]+'LR/'+name[28:-4]+'x4.png'
        # lr_name = name[:18] + 'LR_4x/' + name[21:]
        #input = load_img(self.lr_image_filenames[index]) # type:PIL.Image
        #target=np.asarray(input.getdata()).reshape(3,input.size[0],input.size[1])

        # target = ImageOps.equalize(target)
        # input_eq = ImageOps.equalize(input)
        # target = ImageOps.equalize(target)
        #         # input = ImageOps.equalize(input)
        #input, target, = get_patch(input, target, self.patch_size, self.upscale_factor)

        #if self.data_augmentation:
        #    img_in, img_tar, _ = augment(input, target)

        #if self.transform:
        #    img_in = self.transform(img_in)
        #    img_tar = self.transform(img_tar)

        #return img_in, img_tar
        input, target, = get_patch(self.low[index], self.high[index], self.patch_size, self.upscale_factor)
        if self.data_augmentation:
            img_in, img_tar, _ = augment(input, target)

        if self.transform:
            img_in = self.transform(img_in)
            img_tar = self.transform(img_tar)
        return img_in,img_tar

    def __len__(self):
        #return len(self.hr_image_filenames)
        return len(self.lr_image_filenames)


class DatasetFromFolderEval(data.Dataset):
    def __init__(self, lr_dir, upscale_factor, transform=None):
        super(DatasetFromFolderEval, self).__init__()
        self.image_filenames = [join(lr_dir, x) for x in listdir(lr_dir) if is_image_file(x)]
        self.upscale_factor = upscale_factor
        self.transform = transform

    def __getitem__(self, index):
        input = load_img(self.image_filenames[index])
        _, file = os.path.split(self.image_filenames[index])

        bicubic = rescale_img(input, self.upscale_factor)

        if self.transform:
            # input = self.transform(input)
            bicubic = self.transform(bicubic)

        return bicubic, file

    def __len__(self):
        return len(self.image_filenames)


class Lowlight_DatasetFromVOC(data.Dataset):
    def __init__(self, patch_size, upscale_factor, data_augmentation,
                 transform=None):
        super(Lowlight_DatasetFromVOC, self).__init__()
        self.imgFolder = "datasets/VOC2007/JPEGImages"
        self.image_filenames = [join(self.imgFolder, x) for x in listdir(self.imgFolder) if is_image_file(x)]

        self.image_filenames = self.image_filenames
        self.patch_size = patch_size
        self.upscale_factor = upscale_factor
        self.transform = transform
        self.data_augmentation = data_augmentation

    def __getitem__(self, index):

        ori_img = load_img(self.image_filenames[index])  # PIL image
        width, height = ori_img.size
        ratio = min(width, height) / 384

        newWidth = int(width / ratio)
        newHeight = int(height / ratio)
        ori_img = ori_img.resize((newWidth, newHeight), Image.ANTIALIAS)

        high_image = ori_img

        ## color and contrast *dim*
        color_dim_factor = 0.3 * random.random() + 0.7
        contrast_dim_factor = 0.3 * random.random() + 0.7
        ori_img = ImageEnhance.Color(ori_img).enhance(color_dim_factor)
        ori_img = ImageEnhance.Contrast(ori_img).enhance(contrast_dim_factor)

        ori_img = cv2.cvtColor((np.asarray(ori_img)), cv2.COLOR_RGB2BGR)  # cv2 image
        ori_img = (ori_img.clip(0, 255)).astype("uint8")
        low_img = ori_img.astype('double') / 255.0

        # generate low-light image
        beta = 0.5 * random.random() + 0.5
        alpha = 0.1 * random.random() + 0.9
        gamma = 3.5 * random.random() + 1.5
        low_img = beta * np.power(alpha * low_img, gamma)

        low_img = low_img * 255.0
        low_img = (low_img.clip(0, 255)).astype("uint8")
        low_img = Image.fromarray(cv2.cvtColor(low_img, cv2.COLOR_BGR2RGB))

        img_in, img_tar = get_patch(low_img, high_image, self.patch_size, self.upscale_factor) #从两个图像中的相同位置裁剪出patch_size大小的正方图像

        if self.data_augmentation: #数据增强：随机翻转
            img_in, img_tar, _ = augment(img_in, img_tar)

        if self.transform:
            img_in = self.transform(img_in)
            img_tar = self.transform(img_tar)

        return img_in, img_tar

    def __len__(self):
        return len(self.image_filenames)


class Lowlight_DatasetFromVOC_AlltoMemory(data.Dataset):
    def __init__(self, patch_size, upscale_factor, data_augmentation,
                 transform=None):
        super(Lowlight_DatasetFromVOC_AlltoMemory, self).__init__()
        self.imgFolder = "datasets/VOC2007/JPEGImages"
        self.image_filenames = [join(self.imgFolder, x) for x in listdir(self.imgFolder) if is_image_file(x)]

        self.image_filenames = self.image_filenames
        self.patch_size = patch_size
        self.upscale_factor = upscale_factor
        self.transform = transform
        self.data_augmentation = data_augmentation

        self.low=[]
        self.high=[]
        print("loading dataset to memory")
        for idx in range(len(self.image_filenames)):
            ori_img = load_img(self.image_filenames[idx])  # PIL image
            width, height = ori_img.size
            ratio = min(width, height) / 384

            newWidth = int(width / ratio)
            newHeight = int(height / ratio)
            ori_img = ori_img.resize((newWidth, newHeight), Image.ANTIALIAS)

            high_image = ori_img

            ## color and contrast *dim*
            color_dim_factor = 0.3 * random.random() + 0.7
            contrast_dim_factor = 0.3 * random.random() + 0.7
            ori_img = ImageEnhance.Color(ori_img).enhance(color_dim_factor)
            ori_img = ImageEnhance.Contrast(ori_img).enhance(contrast_dim_factor)

            ori_img = cv2.cvtColor((np.asarray(ori_img)), cv2.COLOR_RGB2BGR)  # cv2 image
            ori_img = (ori_img.clip(0, 255)).astype("uint8")
            low_img = ori_img.astype('double') / 255.0

            # generate low-light image
            beta = 0.5 * random.random() + 0.5
            alpha = 0.1 * random.random() + 0.9
            gamma = 3.5 * random.random() + 1.5
            low_img = beta * np.power(alpha * low_img, gamma)

            low_img = low_img * 255.0
            low_img = (low_img.clip(0, 255)).astype("uint8")
            low_img = Image.fromarray(cv2.cvtColor(low_img, cv2.COLOR_BGR2RGB))

            img_in, img_tar = get_patch(low_img, high_image, self.patch_size, self.upscale_factor)

            if self.data_augmentation:
                img_in, img_tar, _ = augment(img_in, img_tar)

            if self.transform:
                img_in = self.transform(img_in)
                img_tar = self.transform(img_tar)
            self.low.append(img_in)
            self.high.append(img_tar)
            if idx%100==0:
                print("{}/{}".format(idx,len(self.image_filenames)))


    def __getitem__(self, index):
        return self.low[index], self.high[index]

    def __len__(self):
        return len(self.image_filenames)


class Lowlight_DatasetFromVOC_Store(data.Dataset):
    def __init__(self, patch_size, upscale_factor, data_augmentation,
                 transform=None):
        super(Lowlight_DatasetFromVOC, self).__init__()
        self.imgFolder = "datasets/VOC2007/JPEGImages"
        self.image_filenames = [join(self.imgFolder, x) for x in listdir(self.imgFolder) if is_image_file(x)]

        self.image_filenames = self.image_filenames
        self.patch_size = patch_size
        self.upscale_factor = upscale_factor
        self.transform = transform
        self.data_augmentation = data_augmentation

        self.low=[]
        self.high=[]

    def __getitem__(self, index):

        ori_img = load_img(self.image_filenames[index])  # PIL image
        width, height = ori_img.size
        ratio = min(width, height) / 384

        newWidth = int(width / ratio)
        newHeight = int(height / ratio)
        ori_img = ori_img.resize((newWidth, newHeight), Image.ANTIALIAS)

        high_image = ori_img

        ## color and contrast *dim*
        color_dim_factor = 0.3 * random.random() + 0.7
        contrast_dim_factor = 0.3 * random.random() + 0.7
        ori_img = ImageEnhance.Color(ori_img).enhance(color_dim_factor)
        ori_img = ImageEnhance.Contrast(ori_img).enhance(contrast_dim_factor)

        ori_img = cv2.cvtColor((np.asarray(ori_img)), cv2.COLOR_RGB2BGR)  # cv2 image
        ori_img = (ori_img.clip(0, 255)).astype("uint8")
        low_img = ori_img.astype('double') / 255.0

        # generate low-light image
        beta = 0.5 * random.random() + 0.5
        alpha = 0.1 * random.random() + 0.9
        gamma = 3.5 * random.random() + 1.5
        low_img = beta * np.power(alpha * low_img, gamma)

        low_img = low_img * 255.0
        low_img = (low_img.clip(0, 255)).astype("uint8")
        low_img = Image.fromarray(cv2.cvtColor(low_img, cv2.COLOR_BGR2RGB))

        img_in, img_tar = get_patch(low_img, high_image, self.patch_size, self.upscale_factor) #从两个图像中的相同位置裁剪出patch_size大小的正方图像

        if self.data_augmentation: #数据增强：随机翻转
            img_in, img_tar, _ = augment(img_in, img_tar)

        if self.transform:
            img_in = self.transform(img_in)
            img_tar = self.transform(img_tar)

        return img_in, img_tar

    def __len__(self):
        return len(self.image_filenames)

class LowLightDatasetFromFolder(data.Dataset):
    def __init__(self, LR_dir, patch_size, data_augmentation, data_transform):
        self.lr_image_filenames = [join(LR_dir, x) for x in listdir(LR_dir) if is_image_file(x)]
        self.patch_size = patch_size
        self.data_augmentation = data_augmentation
        self.data_transform= data_transform
        self.low = []
        self.tensor=[]
        self.augmentation=torchvision.transforms.Compose([torchvision.transforms.CenterCrop(patch_size),
                                                     torchvision.transforms.RandomHorizontalFlip(0.5),
                                                     torchvision.transforms.RandomVerticalFlip(0.5)])

        self.transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        for idx in range(len(self.lr_image_filenames)):
            input = Image.open(self.lr_image_filenames[idx])  # type:PIL.Image
            array=np.asarray(input).transpose(2,0,1)
            #intensor=self.transform(input)
            self.low.append(input)
            if idx % 50 == 0:
                print("loading:{}/{}".format(idx, len(self.lr_image_filenames)))

    def __getitem__(self, index):
        input=self.low[index]
        #_,x,y=intensor.size()
        #xc = random.randint(0, x - self.patch_size)
        #yc = random.randint(0, y - self.patch_size)
        if self.data_augmentation:
            input = self.augmentation(input)
        if self.data_transform:
            croppedtensor=self.transform(input)
        return croppedtensor

    def __len__(self):
        # return len(self.hr_image_filenames)
        return len(self.lr_image_filenames)