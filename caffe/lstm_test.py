#coding=utf-8
# lstm测试，发现lstm对于图像的像素“模式”很敏感，
#		这里只有使用黑地白子的图片才能识别
# 		似乎必须归一化后才能识别
# 输出发现fc与softmax输出是相同的，这倒是期望的

import sys
sys.path.insert(0,'/home/lbk/deeplab-v2-master/python')
import caffe
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import cv2


path_1 = '/home/lbk/ocr/LSTM/caffe/'

# load net
#net = caffe.Net(path_1+'VGG_VOC2012ext.prototxt', path_1+'VGG_VOC2012ext.caffemodel', caffe.TEST)
net = caffe.Net(path_1+'mnist_test.prototxt', path_1+'lenet_lstm_iter_20000.caffemodel', caffe.TEST)

print "***************************************blob:"
for layer_name,blob in net.blobs.iteritems():
    print layer_name+'\t'+str(blob.data.shape)+'\n'

print "***************************************params:"
for layer_name,param in net.params.iteritems():
    print layer_name+'\t'+str(param[0].data.shape)+'\n'


# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
# shape for input (data blob is N x C x H x W), set data
# run net and take argmax for prediction
img_path = '/home/lbk/ocr/LSTM/caffe/test_image/'

for i in os.listdir(img_path):
	print "test:",i
	file = img_path + i
	# 由于是训练的是单通道图片，所以这里使用单通道的测试图片
	image_src = cv2.imread(file,0)
	image = cv2.resize(image_src,(28,28))

	#cv2.imshow('src',image)
	#cv2.waitKey(0)

	image = np.array(image, dtype=np.float32)
	image = image * 0.00390625
	image = image.transpose(2,0,1)
	image = image[np.newaxis,:,:,:]

	print('image : ',image.shape)
	# shape for input (data blob is N x C x H x W), set data
	#net.blobs['data'].reshape(1, *image.shape)
	net.blobs['data'].data[...] = image
	net.forward()

	fc_out = net.blobs['fc'].data[0]
	softmax_out = net.blobs['prob'].data[0]

	print "predict:",np.argmax(fc_out),np.argmax(softmax_out)