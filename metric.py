import argparse
import time
import os
import logging
import sys

import matlab.engine
import csv

parser = argparse.ArgumentParser()

parser.add_argument('--images', default='image/', help='Enhanced Images')
parser.add_argument('--save', default='EXP/', help='Default folder')
parser.add_argument('--debug', type=bool, default=False)
opt = parser.parse_args()

opt.save = opt.save + '/' + 'Metric-{}'.format(time.strftime("%Y%m%d-%H%M%S"))
os.makedirs(opt.save, exist_ok=True)
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(opt.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

#读取文件夹下所有RGB图片，输出NIQE

eng = matlab.engine.start_matlab()
file = open(opt.save + '/result.csv', mode="w", newline="")
csvf = csv.writer(file)
csvf.writerow(["name", "", "NIQE", ""])

test_list = [os.path.join(opt.images, x) for x in sorted(os.listdir(opt.images))]
for i in test_list:
    eng.


