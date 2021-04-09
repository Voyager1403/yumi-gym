import gym, yumi_gym
import pybullet as p
import numpy as np
import h5py
import time
import os


hf = h5py.File('inference.h5', 'r')
group1 = hf.get('group1')
l_joint_angle = group1.get('l_joint_angle_2')
r_joint_angle = group1.get('r_joint_angle_2')
l_hand_angle = group1.get('l_glove_angle_2')
r_hand_angle = group1.get('r_glove_angle_2')
total_frames = l_joint_angle.shape[0]

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
        action = l_joint_angle[t].tolist() + r_joint_angle[t].tolist() + l_hand_angle[t].tolist() + r_hand_angle[t].tolist()
        # action[5], action[12], action[6], action[13] = 0, 0, 0, 0
        # print(action)
        observation, reward, done, info = env.step(action)
        time.sleep(0.1)

env.close()