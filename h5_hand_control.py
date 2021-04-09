import gym, yumi_gym
import pybullet as p
import numpy as np
import h5py
import time
from math import pi

def linear_map(x_, min_, max_, min_hat, max_hat):
    
    x_hat = 1.0 * (x_ - min_) / (max_ - min_) * (max_hat - min_hat) + min_hat
    print(x_, x_hat, min_, max_, min_hat, max_hat)
    return x_hat

def map_glove_to_inspire_hand(glove_angles):

    ### This function linearly maps the Wiseglove angle measurement to Inspire hand's joint angles.

    ## preparation, specify the range for linear scaling
    hand_start = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.4, 0.0, 0.0]) # radius already
    hand_final = np.array([-1.6, -1.7, -1.6, -1.7, -1.6, -1.7, -1.6, -1.7, -1.0, 0.0, -0.4, -1.0])
    glove_start = np.array([0, 0, 53, 0, 0, 22, 0, 0, 22, 0, 0, 35, 0, 0])# * pi / 180.0 # degree to radius
    glove_final = np.array([45, 100, 0, 90, 120, 0, 90, 120, 0, 90, 120, 0, 90, 120])# * pi / 180.0
    length = glove_angles.shape[0]
    hand_angles = np.zeros((length, 12)) # 12 joints

    ## Iterate to map angles
    for i in range(length):
        # four fingers' extension/flexion (abduction/adduction are dumped)
        hand_angles[i, 0] = linear_map(glove_angles[i, 3], glove_start[3], glove_final[3], hand_start[0], hand_final[0]) # Link1 (joint name)
        hand_angles[i, 1] = linear_map(glove_angles[i, 4], glove_start[4], glove_final[4], hand_start[1], hand_final[1]) # Link11
        hand_angles[i, 2] = linear_map(glove_angles[i, 6], glove_start[6], glove_final[6], hand_start[2], hand_final[2]) # Link2
        hand_angles[i, 3] = linear_map(glove_angles[i, 7], glove_start[7], glove_final[7], hand_start[3], hand_final[3]) # Link22
        hand_angles[i, 4] = linear_map(glove_angles[i, 9], glove_start[9], glove_final[9], hand_start[4], hand_final[4]) # Link3
        hand_angles[i, 5] = linear_map(glove_angles[i, 10], glove_start[10], glove_final[10], hand_start[5], hand_final[5]) # Link33
        hand_angles[i, 6] = linear_map(glove_angles[i, 12], glove_start[12], glove_final[12], hand_start[6], hand_final[6]) # Link4
        hand_angles[i, 7] = linear_map(glove_angles[i, 13], glove_start[13], glove_final[13], hand_start[7], hand_final[7]) # Link44

        # thumb
        hand_angles[i, 8] = (hand_start[8] + hand_final[8]) / 2.0 # Link5 (rotation about z axis), fixed!
        hand_angles[i, 9] = linear_map(glove_angles[i, 2], glove_start[2], glove_final[2], hand_start[9], hand_final[9]) # Link 51
        hand_angles[i, 10] = linear_map(glove_angles[i, 0], glove_start[0], glove_final[0], hand_start[10], hand_final[10]) # Link 52
        hand_angles[i, 11] = linear_map(glove_angles[i, 1], glove_start[1], glove_final[1], hand_start[11], hand_final[11]) # Link 53

    return hand_angles


hf = h5py.File('yumi_intro_YuMi.h5', 'r')
key = '会-hui'
l_glove_angle = hf[key + '/l_glove_angle'][:]
r_glove_angle = hf[key + '/r_glove_angle'][:]
l_hand_angle = map_glove_to_inspire_hand(l_glove_angle)
r_hand_angle = map_glove_to_inspire_hand(r_glove_angle)
l_joint_default = [-1.341905951499939, -1.7764934301376343, 0.5122540473937988, -0.315954327583313, 1.8956027030944824, 0.47532641887664795, -0.7984092235565186]
r_joint_default = [1.2869668006896973, -1.7874925136566162, -0.581124496459961, -0.5961062908172607, -0.1543283462524414, 0.47532641887664795, -0.7984092235565186]
total_frames = l_hand_angle.shape[0]

env = gym.make('yumi-v0')
env.render()
observation = env.reset()

while True:
    env.render()
    for t in range(total_frames):
        # t = 100
        # if t < 30 or t > 180:
        #     continue
        print(t, l_hand_angle.shape)

        # action = [0 for i in range(14)] + l_hand_angle[t].tolist() + r_hand_angle[t].tolist()
        # action[5], action[12], action[6], action[13] = 0, 0, 0, 0
        action = l_joint_default + r_joint_default + l_hand_angle[t].tolist() + r_hand_angle[t].tolist()
        # print(action)
        print(l_hand_angle[t], r_glove_angle[t])
        observation, reward, done, info = env.step(action)
        time.sleep(0.1)