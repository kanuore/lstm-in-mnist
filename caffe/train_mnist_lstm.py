#coding=utf-8
# 加载vgg-5模型，v1_prototxt,只取到9《-64计算loss,并且使用整体loss不除以count
import sys
sys.path.insert(0,'/home/lbk/deeplab-v2-master/python')
import caffe

import numpy as np
import os

#weights = '/home/lbk/segaware/scripts/emb.caffemodel'

# init
#caffe.set_device(int(sys.argv[1]))
caffe.set_mode_gpu()

solver = caffe.SGDSolver('solver.prototxt')
#solver.net.copy_from(weights)
#solver.restore(weights)

# scoring
#val = np.loadtxt('../data/pascal/VOC2010/ImageSets/Main/val.txt', dtype=str)
# 还是以这个循环次数为准

for _ in range(10):
    solver.step(2000)
    #score.seg_tests(solver, False, val, layer='score')
