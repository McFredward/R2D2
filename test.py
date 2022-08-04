import os
import random
import multiprocessing as mp
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from model import Network
from environment import create_env
import config
import ray
import time
import sys
import argparse
device = torch.device('cpu')
torch.set_num_threads(4)

def test(game_name=config.game_name, save_interval = config.save_interval, test_epsilon=config.test_epsilon,
        show=False, save_plot=config.save_plot):

    env = create_env(noop_start=True, clip_rewards=False)
    test_round = 5
    pool = mp.Pool(test_round)
    x1, x2, y = [], [], []

    network = Network(env.action_space.n)
    network.to(device)
    network.share_memory()
    checkpoint = 0
    
    while os.path.exists(f'./models/{game_name}{checkpoint}.pth'):
        state_dict, training_steps, env_steps = torch.load(f'./models/{game_name}{checkpoint}.pth')
        x1.append(training_steps)
        x2.append(env_steps)
        network.load_state_dict(state_dict)

        args = [(network, env) for _ in range(test_round)]
        rewards = pool.map(test_one_case, args)

        print(' training_steps: {}' .format(training_steps))
        print(' env_steps: {}' .format(env_steps))
        print(' average reward: {}\n' .format(sum(rewards)/test_round))
        y.append(sum(rewards)/test_round)
        checkpoint += 1
    
    plt.figure(figsize=(12, 6))
    plt.title(game_name)

    plt.subplot(1, 2, 1)
    plt.xlabel('training steps')
    plt.ylabel('average reward')
    plt.plot(x1, y)

    plt.subplot(1, 2, 2)
    plt.xlabel('environment steps')
    plt.ylabel('average reward')
    plt.plot(x2, y)

    plt.show()
    
    if save_plot:
        plt.savefig('./{}.jpg'.format(game_name))

def test_one_case(args):
    network, env = args
    obs = env.reset()
    network.reset()
    done = False
    obs_history = deque([obs for _ in range(config.frame_stack)], maxlen=config.frame_stack)
    last_action = torch.zeros((1, env.action_space.n))
    sum_reward = 0
    while not done:

        obs = np.stack(obs_history).astype(np.float32)
        obs = torch.from_numpy(obs).unsqueeze(0)
        obs = obs / 255
        action, _, _ = network.step(obs, last_action)

        if random.random() < config.test_epsilon:
            action = env.action_space.sample()

        next_obs, reward, done, _ = env.step(action)
        # print(next_obs)
        obs_history.append(next_obs)
        last_action.fill_(0)
        last_action[0, action] = 1
        sum_reward += reward

    return sum_reward

@ray.remote(num_cpus=1)
def play(checkpoint,args,num_done,rounds=10,client_args="",host=False,port=5060): #-1 for the last snapshot

    env_name = args.env_name
    render = config.render

    if "CartPole" in env_name:
        num_player = 0
        env = create_env(env_name=env_name, clip_rewards=False, testing=True, multi_conf=client_args)
    else:
        num_player = int(checkpoint.split('.')[0][-1])
        env = create_env(env_name=args.env_name, clip_rewards=False,testing=True,multi_conf=client_args,is_host=host,port=port,num_players=args.num_player,name='Player_'+str(num_player))

    network = Network(env.action_space.n)
    network.to(device)
    network.share_memory()

    #file = directory+"/Vizdoom" + str(checkpoint) + ".pth"
    if torch.cuda.is_available():
        state_dict, _, _ = torch.load(checkpoint)
    else:
        state_dict, _, _ = torch.load(checkpoint, map_location=torch.device('cpu'))

    network.load_state_dict(state_dict)
    print("Loaded "+str(args.file.split('/')[-1]))
    # ---Test trained network---
    sum_reward = 0
    for i in range(rounds):
        reward = test_one_case((network,env))
        print("reward P{} = {:.3f}".format(num_player+1,reward))
        sum_reward += reward
        if render:
            env.reset()
            for t in range(200):
                env.render()
    print("mean reward = {:.3f}".format(sum_reward / rounds))
    num_done[0] += 1


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", dest='file',type=str)
    parser.add_argument('--env_name', dest='env_name', default=config.game_name+config.env_type)
    parser.add_argument("--multiplayer", action='store_true', default=False)
    parser.add_argument("--num_player", dest='num_player', default=-1)
    parser.add_argument("--num_rounds", dest="num_rounds", type=int, default=30)
    args = parser.parse_args()

    num_done = [0]
    if not args.multiplayer:
        play.remote(args.file,args,num_done,args.num_rounds)

        while num_done[0] < 1:
            time.sleep(config.log_interval)

    else:
        checkpoints = [os.path.join(args.file, filename) for filename in os.listdir(args.file) if os.path.isfile(os.path.join(args.file, filename)) and filename.split('.')[-1] == 'pth']
        if args.num_player == -1:
            args.num_player = len(checkpoints)

        play.remote(checkpoints[0],args,num_done,10000,client_args="",host=True,port=5060)
        for num_player in range(1,args.num_player):
            play.remote(checkpoints[num_player],args,num_done,10000,client_args="127.0.0.1:5060",host=False)

        while num_done[0] < args.num_player:
            time.sleep(config.log_interval)


