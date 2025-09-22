import argparse
import configparser
import copy
import logging
import os
import pickle
import random
from collections import deque
from math import pow

import gym
import h5py
import numpy as np
import torch
from numpy.linalg import norm
from risk import risk

from crowd_sim.envs.policy.orca import ORCA
from crowd_sim.envs.utils.action import ActionXY
from crowd_sim.envs.utils.info import *
# from crowd_nav.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.state import JointState


def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--env_config', type=str,
                        default='configs/env.config')
    parser.add_argument('--policy_config', type=str,
                        default='configs/env.config')
    parser.add_argument('--policy', type=str, default='orca')
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--mode', type=str, default='d')
    args = parser.parse_args()

    env_config_file = args.env_config
    policy_config_file = args.policy_config

    # configure logging and device
    logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info('Using device: %s', device)

    # configure policy

    policy = ORCA()
    policy_config = configparser.RawConfigParser()
    policy_config.read(policy_config_file)
    policy.configure(policy_config)
    if args.policy == 'orca':
        policy.safety_space = 0.05

    # configure environment
    env_config = configparser.RawConfigParser()
    env_config.read(env_config_file)
    env = gym.make('CrowdSim-v0')
    env.configure(env_config)

    robot = Robot(env_config, 'robot')
    robot.set_policy(policy)
    env.set_robot(robot)

    policy.set_phase(args.phase)
    policy.set_device(device)

    policy.set_env(env)
    robot.policy.set_phase(args.phase)
    robot.print_info()

    random.seed(42)
    
    capacity = 500000
    num_query = 200
    traj_len = 15

    run = True
    observations = []
    actions = []
    rewards = []
    terminals = []
    timeouts = []
    too_close = 0
    next_observations = []

    indices_1 = []
    indices_2 = []
    human_labels = []

    dis_list = []
    danger_list = []

    

    success = 0
    total = 0
    collision = 0
    timeout = 0
    step = 0
    index = -1
    pre_info = None

    pre_dis_list = []
    pre_danger_list = []

    now_dis_list = []
    now_danger_list = []
    while run:
        j = 0
        ob = env.reset(phase=args.phase)
        deque_list = deque(maxlen=8)
        deque_list.clear()
        index += 1
        danger = 0
        done = False
        while not done:
            action_xy = robot.act(ob)
            vx = action_xy.vx + np.random.normal(0, 0.1)
            vx = clamp_xy(vx, -1, 1)
            vy = action_xy.vy + np.random.normal(0, 0.1)
            vy = clamp_xy(vy, -1, 1)
            action_xy = ActionXY(vx, vy)
            joint_state = JointState(robot.get_full_state(), ob)
            state = to_np(robot.policy.transform(joint_state).view(1, -1).squeeze(0))
            ob, reward, done, info = env.step(action_xy)
            now_dis_list.append(norm(np.array(robot.get_position()) - np.array(robot.get_goal_position())))
            if isinstance(info, Danger):
                danger += 1
            new_joint_state = JointState(robot.get_full_state(), ob)
            next_state = to_np(robot.policy.transform(new_joint_state).view(1, -1).squeeze(0))
            action = np.array([action_xy.vx, action_xy.vy])
            # env.render(mode='video')
            observations.append(state)
            next_observations.append(next_state)
            actions.append(action)
            rewards.append(reward)
            terminals.append(int(done))
            j += 1
            step += 1

            if isinstance(info, Timeout):
                timeouts.append(1)
            else:
                timeouts.append(0)

            if isinstance(info, Danger):
                too_close += 1
                now_danger_list.append(1)
            else:
                now_danger_list.append(0)

            if len(observations) == capacity:
                run = False
                break
        if len(human_labels) < num_query and index % 5 == 0:
            pre_info = info
            pre_dis_list = copy.deepcopy(now_dis_list)
            pre_danger_list = copy.deepcopy(now_danger_list)
        if len(human_labels) < num_query and index % 5 == 1 and len(pre_dis_list) >= traj_len and len(now_dis_list) >= traj_len:
            l1 = len(pre_dis_list)
            l2 = len(now_dis_list)
            if isinstance(pre_info, Collision) and isinstance(info, Collision):
                if pre_dis_list[l1 - 1] < now_dis_list[l2 - 1]:
                    human_labels.append(0.7)
                elif pre_dis_list[l1 - 1] > now_dis_list[l2 - 1]:
                    human_labels.append(0.3)
                else:
                    human_labels.append(0.5)
                indices_1.append(step - j - traj_len)
                indices_2.append(step - traj_len)
                danger_list.append(sum(pre_danger_list[l1 - traj_len: l1]))
                danger_list.append(sum(now_danger_list[l2 - traj_len: l2]))
                dis_list.append(pre_dis_list[l1 - traj_len] - pre_dis_list[l1 - 1])
                dis_list.append(now_dis_list[l2 - traj_len] - now_dis_list[l2 - 1])
            elif isinstance(pre_info, Collision):
                s2 = random.randint(0, l2 - traj_len)
                human_labels.append(0.1)
                indices_1.append(step - j - traj_len)
                indices_2.append(step - (l2 - s2))
                danger_list.append(sum(pre_danger_list[l1 - traj_len: l1]))
                danger_list.append(sum(now_danger_list[s2:s2 + traj_len]))
                dis_list.append(pre_dis_list[l1 - traj_len] - pre_dis_list[l1 - 1])
                dis_list.append(now_dis_list[s2] - now_dis_list[s2 + traj_len - 1])
            elif isinstance(info, Collision):
                s1 = random.randint(0, l1 - traj_len)
                human_labels.append(0.9)
                indices_1.append(step - j - (l1 - s1))
                indices_2.append(step - traj_len)
                danger_list.append(sum(pre_danger_list[s1: s1 + traj_len]))
                danger_list.append(sum(now_danger_list[l2 - traj_len:l2]))
                dis_list.append(pre_dis_list[s1] - pre_dis_list[s1 + traj_len - 1])
                dis_list.append(now_dis_list[l2 - traj_len] - now_dis_list[l2 - 1])
            else:
                s1 = random.randint(0, l1 - traj_len)
                s2 = random.randint(0, l2 - traj_len)
                if pre_dis_list[s1] - pre_dis_list[s1 + traj_len - 1] > now_dis_list[s2] - now_dis_list[
                    s2 + traj_len - 1]:
                    human_labels.append(0.7)
                elif pre_dis_list[s1] - pre_dis_list[s1 + traj_len - 1] < now_dis_list[s2] - now_dis_list[
                    s2 + traj_len - 1]:
                    human_labels.append(0.3)
                else:
                    human_labels.append(0.5)
                danger_list.append(sum(pre_danger_list[s1: s1 + traj_len]))
                danger_list.append(sum(now_danger_list[s2:s2 + traj_len]))
                dis_list.append(pre_dis_list[s1] - pre_dis_list[s1 + traj_len - 1])
                dis_list.append(now_dis_list[s2] - now_dis_list[s2 + traj_len - 1])
                indices_1.append(step - j - (l1 - s1))
                indices_2.append(step - (l2 - s2))
            pre_dis_list.clear()
            pre_danger_list.clear()
        now_dis_list.clear()
        now_danger_list.clear()

        if isinstance(info, ReachGoal):
            success += 1
            total += 1
        elif isinstance(info, Collision):
            collision += 1
            total += 1
        elif isinstance(info, Timeout):
            timeout += 1
            total += 1

    print(len(human_labels))

    path = "data"
    human_indices_1_file = "indices_1.pkl"
    human_indices_2_file = "indices_2.pkl"
    human_labels_file = "human_labels.pkl"
    if not os.path.exists(path):
        os.makedirs(path)
    file_path = os.path.join(path, human_indices_1_file)
    with open(file_path, "wb") as fp:
        pickle.dump(indices_1, fp)
    file_path = os.path.join(path, human_indices_2_file)
    with open(file_path, "wb") as fp:
        pickle.dump(indices_2, fp)
    file_path = os.path.join(path, human_labels_file)
    with open(file_path, "wb") as fp:
        pickle.dump(human_labels, fp)

    dis_design = os.path.join(path, "dis.pkl")
    danger_design = os.path.join(path, "danger.pkl")
    with open(dis_design, "wb") as fp:
        pickle.dump(dis_list, fp)
    with open(danger_design, "wb") as fp:
        pickle.dump(danger_list, fp)


    file = 'dataset'
    file_name = '{}/{}'.format(file, file) + '.hdf5'

    if not os.path.exists(file):
        print('start make data')
        os.mkdir(file)
        f = h5py.File(file_name, "w")
        f.create_dataset("observations", data=observations)
        f.create_dataset("next_observations", data=next_observations)
        f.create_dataset("actions", data=actions)
        f.create_dataset("rewards", data=np.array(rewards))
        f.create_dataset("terminals", data=np.array(terminals))
        f.create_dataset("timeouts", data=np.array(timeouts))
        f.close()
        if os.path.exists(file_name):
            print('make success')
        else:
            print('make fail')
    else:
        print('finish')

    risk()


def avg(input_list):
    if input_list:
        return sum(input_list) / len(input_list)
    else:
        return


def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()


def clamp_xy(num, min, max):
    if num < min:
        num = min
    elif num > max:
        num = max
    else:
        pass
    return num


if __name__ == '__main__':
    main()
