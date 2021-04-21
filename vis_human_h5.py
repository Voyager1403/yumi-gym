import h5py
import math
from scipy.spatial.transform import Rotation as R
import torch
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np


def parse_h5(filename):
    # data_list = []
    h5_file = h5py.File(filename, 'r')
    # print(filename, h5_file.keys())
    for key in h5_file.keys():
        data_list = []
        print(key)
        # position data
        l_shoulder_pos = h5_file[key + '/l_up_pos'][:]
        r_shoulder_pos = h5_file[key + '/r_up_pos'][:]
        l_elbow_pos = h5_file[key + '/l_fr_pos'][:]
        r_elbow_pos = h5_file[key + '/r_fr_pos'][:]
        l_wrist_pos = h5_file[key + '/l_hd_pos'][:]
        r_wrist_pos = h5_file[key + '/r_hd_pos'][:]
        # quaternion data
        l_shoulder_quat = R.from_quat(h5_file[key + '/l_up_quat'][:])
        r_shoulder_quat = R.from_quat(h5_file[key + '/r_up_quat'][:])
        l_elbow_quat = R.from_quat(h5_file[key + '/l_fr_quat'][:])
        r_elbow_quat = R.from_quat(h5_file[key + '/r_fr_quat'][:])
        l_wrist_quat = R.from_quat(h5_file[key + '/l_hd_quat'][:])
        r_wrist_quat = R.from_quat(h5_file[key + '/r_hd_quat'][:])
        # euler data
        l_shoulder_euler = l_shoulder_quat.as_euler('xyz', degrees=True)
        r_shoulder_euler = r_shoulder_quat.as_euler('xyz', degrees=True)
        l_elbow_euler = l_elbow_quat.as_euler('xyz', degrees=True)
        r_elbow_euler = r_elbow_quat.as_euler('xyz', degrees=True)
        l_wrist_euler = l_wrist_quat.as_euler('xyz', degrees=True)
        r_wrist_euler = r_wrist_quat.as_euler('xyz', degrees=True)
        # print(l_shoulder_pos.shape, r_shoulder_pos.shape, l_elbow_pos.shape, r_elbow_pos.shape, l_wrist_pos.shape, r_wrist_pos.shape)
        
        total_frames = l_shoulder_pos.shape[0]
        for t in range(total_frames):
            # x
            x = torch.stack([torch.from_numpy(l_shoulder_euler[t]),
                             torch.from_numpy(l_elbow_euler[t]),
                             torch.from_numpy(l_wrist_euler[t]),
                             torch.from_numpy(r_shoulder_euler[t]),
                             torch.from_numpy(r_elbow_euler[t]),
                             torch.from_numpy(r_wrist_euler[t])], dim=0)
            # number of nodes
            num_nodes = 6
            # edge index
            edge_index = torch.LongTensor([[0, 1, 3, 4],
                                           [1, 2, 4, 5]])
            # position
            pos = torch.stack([torch.tensor(l_shoulder_pos[t]),
                               torch.tensor(l_elbow_pos[t]),
                               torch.tensor(l_wrist_pos[t]),
                               torch.tensor(r_shoulder_pos[t]),
                               torch.tensor(r_elbow_pos[t]),
                               torch.tensor(r_wrist_pos[t])], dim=0)
            # edge attributes
            edge_attr = []
            for edge in edge_index.permute(1, 0):
                parent = edge[0]
                child = edge[1]
                edge_attr.append(pos[child] - pos[parent])
            edge_attr = torch.stack(edge_attr, dim=0)
            # skeleton type & topology type
            skeleton_type = 0
            topology_type = 0
            # end effector mask
            ee_mask = torch.zeros(6, 1).bool()
            ee_mask[2] = ee_mask[5] = True
            # parent
            parent = torch.LongTensor([-1, 0, 1, -1, 3, 4])
            # offset
            offset = torch.zeros(num_nodes, 3)
            for node_idx in range(num_nodes):
                if parent[node_idx] != -1:
                    offset[node_idx] = pos[node_idx] - pos[parent[node_idx]]
            # distance to root
            root_dist = torch.zeros(num_nodes, 1)
            for node_idx in range(num_nodes):
                dist = 0
                current_idx = node_idx
                while parent[current_idx] != -1:
                    origin = offset[current_idx]
                    offsets_mod = math.sqrt(origin[0]**2+origin[1]**2+origin[2]**2)
                    dist += offsets_mod
                    current_idx = parent[current_idx]
                root_dist[node_idx] = dist
            
            data = Data(x=x,
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                        pos=pos,
                        skeleton_type=skeleton_type,
                        topology_type=topology_type,
                        ee_mask=ee_mask,
                        root_dist=root_dist,
                        num_nodes=num_nodes,
                        parent=parent,
                        offset=offset)
            # print(data)
            data_list.append(data)
        def run(t):
            pos = data_list[t].pos
            edge_index = data_list[t].edge_index
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
        lines = [ax.plot([], [], [], 'royalblue', marker='o')[0] for i in range(data_list[0].edge_index.shape[1])]
        total_frames = len(data_list)
        ani = animation.FuncAnimation(fig, run, np.arange(total_frames), interval=50,repeat=True)
        plt.show()
    return data_list

if __name__ == '__main__':
    # parse_h5(filename='/home/yu/PycharmProjects/MotionTransfer-master-Yu-comment/data/source/sign/h5/total_mocap_data_YuMi.h5')
    parse_h5(filename='/home/yu/PycharmProjects/MotionTransfer-master-Yu-comment/test.h5')
