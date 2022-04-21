# encoding: utf-8
"""
@author: luoyang
@time: 2022/3/4 2:07 PM
@desc: s
"""
from itertools import count

import socket
import struct
import argparse
import os
import numpy as np
import ddpg


def decode_charArray(line):
    val_vec = []
    for i in range(7):
        val_vec.append(struct.unpack('d', line[5 + i * 8: 13 + i * 8])[0])
    return val_vec

def cal_reward(vec):
    v_diff = abs(vec[0]-vec[4])
    h_diff = abs(vec[1]-vec[5])
    isStable = int(vec[6])
    r = - v_diff - h_diff
    return h_diff, v_diff, isStable, r


# parse arguments
parser = argparse.ArgumentParser()
# common used
parser.add_argument('--mode', default='train', type=str)  # mode = 'train' or 'test'
parser.add_argument('--tau', default=0.005, type=float)  # target smoothing coefficient
parser.add_argument('--exp', default="exp-1", type=str)  # experiment times


# uncommon used(by default)
parser.add_argument('--isload', default=False, type=bool)
parser.add_argument('--capacity', default=1000000, type=int)  # replay buffer size
parser.add_argument('--update_iteration', default=200, type=int)
parser.add_argument('--gamma', default=0.99, type=int) # discounted factor
parser.add_argument('--batch_size', default=100, type=int) # mini batch size
parser.add_argument('--exploration_noise', default=0.1, type=float)
args = parser.parse_args()



def main():

    current_path = os.getcwd()
    model_path = os.path.join(current_path,args.exp,"model")
    if not os.path.exists(args.exp) :
        os.mkdir(args.exp)
    if not os.path.exists(model_path) :
        os.mkdir(model_path)

    # SOCK_DGRAM : UDP
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(('127.0.0.1', 8888))
    print('Waiting for connection...')

    # initial ddpg algorithm

    max_action = float(10.0)
    state_dim = 7
    action_dim =2
    agent = ddpg.init(model_path,args.mode,state_dim, action_dim,max_action)
    action_space_low = -20.0
    action_space_high = 20.0

    error_data_count = 0
    state_cache = [[], []]
    action_cache = []
    episode_reward = 0
    try:
        while True:
            data, addr = s.recvfrom(1024)
            # check data
            if data[0:5] != b'DATA0':
                continue
            elif len(data) != 61:
                error_data_count += 1
                continue
            else:
                new_state = np.asarray(decode_charArray(data), dtype=np.float32)
                action = agent.select_action(new_state)
                state_cache[0] = state_cache[1]
                state_cache[1] = new_state
                if args.mode == "train":
                    if state_cache[0] == []:
                        action = (action + np.random.normal(0, args.exploration_noise, size=2)).clip(action_space_low, action_space_high)
                        action_cache = action
                        ls = action.tolist()
                        s.sendto(b'DATA1' + struct.pack("d", ls[0]) + struct.pack("d", ls[1]), addr)

                    else:
                        action = (action + np.random.normal(0, args.exploration_noise, size=2)).clip(action_space_low,
                                                                                                     action_space_high)
                        v_diff, h_diff, isStable, reward = cal_reward(new_state)
                        print(new_state,action)
                        # check state
                        isDone = 1 if v_diff + h_diff < 2 or v_diff > 10 or h_diff > 15 and isStable == 1 else 0
                        agent.replay_buffer.push((state_cache[0], state_cache[1], action_cache, reward, np.float(isDone)))
                        action_cache = action
                        episode_reward += reward
                        #if isDone == 1:
                        #    break

                        ls = action.tolist()
                        s.sendto(b'DATA1'+struct.pack("d", ls[0]) + struct.pack("d", ls[1]), ('127.0.0.1', 8888))
                else:
                    ls = action.tolist()
                    s.sendto(b'DATA1' + struct.pack("d", ls[0]) + struct.pack("d", ls[1]), ('127.0.0.1', 8888))
    except:
        pass
    finally:
        if args.mode == "train" :
            agent.update(args.tau, args.update_iteration, args.batch_size, args.gamma)
            agent.save()
        else :
            pass

    print(f"episode done total reward {episode_reward}")


if __name__ == '__main__':
    main()
