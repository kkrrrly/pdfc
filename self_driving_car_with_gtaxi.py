# encoding: utf-8
"""
@author: luoyang
@time: 2022/4/19 4:01 PM
@desc:
"""

import gym
import os
import argparse
import numpy as np

from ddpg_ori import DDPG
from itertools import count
from functools import reduce


# parse arguments
parser = argparse.ArgumentParser()
# common used
parser.add_argument('--mode', default='train', type=str)  # mode = 'train' or 'test'
parser.add_argument('--tau', default=0.005, type=float)  # target smoothing coefficient
parser.add_argument('--exp', default="exp-2", type=str)  # experiment times

parser.add_argument('--test_iteration', default=10, type=int)


# uncommon used(by default)
parser.add_argument('--isload', default=False, type=bool)

parser.add_argument('--capacity', default=1000000, type=int)  # replay buffer size
parser.add_argument('--update_iteration', default=200, type=int)
parser.add_argument('--gamma', default=0.99, type=int) # discounted factor
parser.add_argument('--batch_size', default=100, type=int) # mini batch size
parser.add_argument('--exploration_noise', default=0.2, type=float)
parser.add_argument('--max_episode', default=100000, type=int) # num of games

# render
parser.add_argument('--render', default=True, type=bool) # show UI or not
parser.add_argument('--render_interval', default=0, type=int)


# save
parser.add_argument('--log_interval', default=500, type=int) #

args = parser.parse_args()

def recover(l_action):
    if  l_action[1] >0:
        return np.array([l_action[0],l_action[1],0.])
    elif l_action[1] < 0:
        return np.array([l_action[0], 0., l_action[1]])
    else:
        return np.array([l_action[0], 0., 0.])

def main():
    env = gym.make("CarRacing-v1")

    state_dim = reduce(lambda x,y : x*y , env.observation_space.shape)
    action_dim = 2
    max_action = 1.0
    #min_Val = torch.tensor(1e-7).float().to(device)  # min value

    current_path = os.getcwd()
    model_path = os.path.join(current_path,args.exp,"model")
    if not os.path.exists(args.exp) :
        os.mkdir(args.exp)
    if not os.path.exists(model_path) :
        os.mkdir(model_path)

    agent = DDPG(state_dim,action_dim,max_action,args.update_iteration,args.batch_size,args.gamma,args.tau, model_path)
    ep_r = 0
    if args.mode == 'test':
        agent.load()
        for i in range(args.test_iteration):
            state = env.reset()
            for t in count():
                action = agent.select_action(state)
                rpc_action = recover(action)
                next_state, reward, done, info = env.step(np.float32(rpc_action))
                ep_r += reward
                env.render()
                if done or t >= args.max_length_of_trajectory:
                    print("Ep_i \t{}, the ep_r is \t{:0.2f}, the step is \t{}".format(i, ep_r, t))
                    ep_r = 0
                    break
                state = next_state

    elif args.mode == 'train':
        if args.isload : agent.load(model_path)
        total_step = 0
        for i in range(args.max_episode):
            total_reward = 0
            step =0
            state = env.reset()
            for t in count():


                action = agent.select_action(state)

                action = (action + np.random.normal(0, args.exploration_noise, size=action_dim)).clip(
                    np.array([-1,-1],float),  np.array([1,1],float))
                rpc_action = recover(action)

                next_state, reward, done, info = env.step(rpc_action)
                if t in range(50):
                    print(f"this episode step {t} {action}  {rpc_action}")
                if args.render and i >= args.render_interval : env.render()
                agent.replay_buffer.push((state, next_state, action, reward, np.float(done)))

                state = next_state
                if total_reward < 0:
                    done = True

                if done:
                    break
                step += 1
                total_reward += reward
            total_step += step+1
            print("Total T:{} Episode: \t{} Total Reward: \t{:0.2f}".format(total_step, i, total_reward))
            agent.update()
           # "Total T: %d Episode Num: %d Episode T: %d Reward: %f

            if i % args.log_interval == 0:
                agent.save(model_path)
    else:
        raise NameError("mode wrong!!!")




if __name__ == '__main__':
    main()