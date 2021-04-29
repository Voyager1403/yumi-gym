import h5py
import math
from scipy.spatial.transform import Rotation as R
import torch
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
import time
from body_movement_vis import pyKinect
import os



edge_index = torch.LongTensor([[0, 2, 1, 3],
                               [2, 4, 3, 5]])
data_list = []
def parse_h5(g1):


    def run(t):
        data_list = []
        kinect_pos, kinect_quat = pyKinect.get_data()

        # position data         shape(Tx3)
        l_shoulder_pos = torch.tensor(kinect_pos[0])
        r_shoulder_pos = torch.tensor(kinect_pos[1])

        origin = (l_shoulder_pos + r_shoulder_pos) / 2  # calculate origin point
        # define rotation matrix
        rot_mat = torch.tensor([[0, 1, 0],
                                [0, 0, -1],
                                [-1, 0, 0]]).float()

        l_shoulder_pos = ((l_shoulder_pos - origin).matmul(rot_mat)) / 1000
        r_shoulder_pos = ((r_shoulder_pos - origin).matmul(rot_mat)) / 1000
        l_elbow_pos    = ((torch.tensor(kinect_pos[2]) - origin).matmul(rot_mat)) / 1000
        r_elbow_pos    = ((torch.tensor(kinect_pos[3]) - origin).matmul(rot_mat)) / 1000
        l_wrist_pos    = ((torch.tensor(kinect_pos[4]) - origin).matmul(rot_mat)) / 1000
        r_wrist_pos    = ((torch.tensor(kinect_pos[5]) - origin).matmul(rot_mat)) / 1000

        data_list.append(l_shoulder_pos.tolist())
        data_list.append(r_shoulder_pos.tolist())
        data_list.append(l_elbow_pos.tolist())
        data_list.append(r_elbow_pos.tolist())
        data_list.append(l_wrist_pos.tolist())
        data_list.append(r_wrist_pos.tolist())

        l_shoulder_quat=[]
        r_shoulder_quat=[]
        l_elbow_quat=[]
        r_elbow_quat=[]
        l_wrist_quat=[]
        r_wrist_quat=[]
        # quaternion data
        l_shoulder_quat.append(kinect_quat[0])
        r_shoulder_quat.append(kinect_quat[1])
        l_elbow_quat.append(kinect_quat[2])
        r_elbow_quat.append(kinect_quat[3])
        l_wrist_quat.append(kinect_quat[4])
        r_wrist_quat.append(kinect_quat[5])

        print('frame ' + str(t))
        # time.sleep(0.1)

        g1["/group1/l_up_pos"][t] = l_shoulder_pos.tolist()
        g1["/group1/r_up_pos"][t] = r_shoulder_pos.tolist()
        g1["/group1/l_fr_pos"][t] = l_elbow_pos.tolist()
        g1["/group1/r_fr_pos"][t] = r_elbow_pos.tolist()
        g1["/group1/l_hd_pos"][t] = l_wrist_pos.tolist()
        g1["/group1/r_hd_pos"][t] = r_wrist_pos.tolist()
        g1["/group1/l_up_quat"][t] = l_shoulder_quat
        g1["/group1/r_up_quat"][t] = r_shoulder_quat
        g1["/group1/l_fr_quat"][t] = l_elbow_quat
        g1["/group1/r_fr_quat"][t] = r_elbow_quat
        g1["/group1/l_hd_quat"][t] = l_wrist_quat
        g1["/group1/r_hd_quat"][t] = r_wrist_quat


        print("write frame "+ str(t))

        pos = torch.tensor(data_list)
        # edge index

        for line, edge in zip(lines, edge_index.permute(1, 0)):
            line_x = [pos[edge[0]][0], pos[edge[1]][0]]
            line_y = [pos[edge[0]][1], pos[edge[1]][1]]
            line_z = [pos[edge[0]][2], pos[edge[1]][2]]
            line.set_data(np.array([line_x, line_y]))
            line.set_3d_properties(np.array(line_z))
        return lines

    # attach 3D axis to figure
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    ax.view_init(elev=0, azim=0)
    # set axis limits & labels
    ax.set_xlim3d([-0.2, 0.2])
    ax.set_xlabel('X')
    ax.set_ylim3d([-0.4, 0.4])
    ax.set_ylabel('Y')
    ax.set_zlim3d([-0.5, 0])
    ax.set_zlabel('Z')
    # create animation
    lines = [ax.plot([], [], [], 'royalblue', marker='o')[0] for i in range(edge_index.shape[1])]
    total_frames = len(data_list)
    ani = animation.FuncAnimation(fig, run, interval=50,frames=frame,repeat=False)
    plt.show()

    return data_list

frame=500

pyKinect.init()
for i in range(5):#录制5段动作
    hf = h5py.File(os.path.join("/home/yu/PycharmProjects/MotionTransfer-master-Yu-comment/", 'kinect_h5/random'+str(i)+'.h5'), 'w')

    g1 = hf.create_group('group1')
    # parse_h5(filename='/home/yu/PycharmProjects/MotionTransfer-master-Yu-comment/data/source/sign/h5/total_mocap_data_YuMi.h5')

    g1.create_dataset(name="/group1/l_up_pos", shape=(frame,3))
    g1.create_dataset(name="/group1/r_up_pos", shape=(frame,3))
    g1.create_dataset(name="/group1/l_fr_pos", shape=(frame,3))
    g1.create_dataset(name="/group1/r_fr_pos", shape=(frame,3))
    g1.create_dataset(name="/group1/l_hd_pos", shape=(frame,3))
    g1.create_dataset(name="/group1/r_hd_pos", shape=(frame,3))
    g1.create_dataset(name="/group1/l_up_quat",shape=(frame,4))
    g1.create_dataset(name="/group1/r_up_quat",shape=(frame,4))
    g1.create_dataset(name="/group1/l_fr_quat",shape=(frame,4))
    g1.create_dataset(name="/group1/r_fr_quat",shape=(frame,4))
    g1.create_dataset(name="/group1/l_hd_quat",shape=(frame,4))
    g1.create_dataset(name="/group1/r_hd_quat",shape=(frame,4))
    g1.create_dataset(name="/group1/l_glove_angle", data=np.zeros((frame,15)))#这里假设group1已经存在
    g1.create_dataset(name="/group1/r_glove_angle", data=np.zeros((frame,15)))

    parse_h5(g1)

pyKinect.deinit()
hf.close()
