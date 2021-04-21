import h5py
import numpy as np
import os


frame=50

#打开文件
hf = h5py.File(os.path.join("/home/yu/PycharmProjects/MotionTransfer-master-Yu-comment/", 'test.h5'), 'a')#a：追加模式

hf.create_dataset(name="/group1/l_glove_angle", data=np.zeros((frame,15)))#这里假设group1已经存在
hf.create_dataset(name="/group1/r_glove_angle", data=np.zeros((frame,15)))

#关闭文件
hf.close()
