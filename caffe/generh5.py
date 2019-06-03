#coding=utf-8
import os
import h5py
import numpy as np

imgData = np.ones((128,28))
imgData[:,0] = 0
print imgData
print imgData.shape


if not os.path.exists('1.h5'):
    with h5py.File('1.h5') as f:
        f['cont'] = imgData # 使用H5文件要注意文件名是否与prototxt对应

f = h5py.File('1.h5','r')   #打开h5文件
print f.keys()                            #可以查看所有的主键
print f['cont'][:]