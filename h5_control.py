import gym, yumi_gym
import pybullet as p
import numpy as np
import h5py
import time
import os

file = './checkpoints/2021-3-21/optimize-wo.h5'
print(file)
hf = h5py.File(file, 'r')
group1 = hf.get('group1')
print(group1.keys())
l_joint_angle = group1.get('l_joint_angle_2')
r_joint_angle = group1.get('r_joint_angle_2')
l_hand_angle = group1.get('l_glove_angle_2')
r_hand_angle = group1.get('r_glove_angle_2')
l_hand_pos = group1.get('l_joint_pos_2')[:, 6]
r_hand_pos = group1.get('r_joint_pos_2')[:, 6]
l_joint_default = [-1.341905951499939, -1.7764934301376343, 0.5122540473937988, -0.315954327583313, 1.8956027030944824, 0.47532641887664795, -0.7984092235565186]
r_joint_default = [1.2869668006896973, -1.7874925136566162, -0.581124496459961, -0.5961062908172607, -0.1543283462524414, 0.47532641887664795, -0.7984092235565186]
height_limit = -0.25
total_frames = l_joint_angle.shape[0]
l_joint_angle_processed = []
r_joint_angle_processed = []

env = gym.make('yumi-v0')
env.render()
observation = env.reset()

while True:
    env.render()
    for t in range(total_frames):
        # t = 100
        # if t < 30 or t > 180:
        #     continue
        print(t, l_joint_angle.shape)

        action = []
        if l_hand_pos[t][2] > height_limit:
            action += l_joint_angle[t].tolist()
            l_joint_angle_processed.append(l_joint_angle[t].tolist())
        else:
            action += l_joint_default
            l_joint_angle_processed.append(l_joint_default)
        if r_hand_pos[t][2] > height_limit:
            action += r_joint_angle[t].tolist()
            r_joint_angle_processed.append(r_joint_angle[t].tolist())
        else:
            action += r_joint_default
            r_joint_angle_processed.append(r_joint_default)
        action += l_hand_angle[t].tolist() + r_hand_angle[t].tolist()
        # action = l_joint_angle[t].tolist() + r_joint_angle[t].tolist() + l_hand_angle[t].tolist() + r_hand_angle[t].tolist()
        # action[5], action[12], action[6], action[13] = 0, 0, 0, 0
        # print(action)
        observation, reward, done, info = env.step(action)
        time.sleep(0.02)

    # l_joint_angle_processed = np.stack(l_joint_angle_processed, axis=0)
    # r_joint_angle_processed = np.stack(r_joint_angle_processed, axis=0)
    # hf_processed = h5py.File('processed.h5', 'w')
    # g1 = hf_processed.create_group('group1')
    # g1.create_dataset('l_joint_angle_2', data=l_joint_angle_processed)
    # g1.create_dataset('r_joint_angle_2', data=r_joint_angle_processed)
    # g1.create_dataset('l_glove_angle_2', data=l_hand_angle)
    # g1.create_dataset('r_glove_angle_2', data=r_hand_angle)
    # print('Processing done!!!')
    # break
env.close()