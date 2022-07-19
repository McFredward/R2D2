import random
import time
import torch
import numpy as np
import ray
from worker import Learner, Actor, ReplayBuffer
import config
from vizdoom import scenarios_path

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.set_num_threads(1)

def get_epsilon(actor_id: int, base_eps: float = config.base_eps, alpha: float = config.alpha, num_actors: int = config.num_actors):
    exponent = 1 + actor_id / (num_actors-1) * alpha
    return base_eps**exponent


def train(num_actors=config.num_actors, log_interval=config.log_interval):
    ray.init()
    num_players = config.num_players if config.multiplayer else 1

    #instance = (buffer,learner,actors)
    instances = []
    for player in range(num_players):

        buffer = ReplayBuffer.remote(player)
        learner = Learner.remote(buffer,config.pretrain)

        multi_conf = ""
        if config.multiplayer:
            #the first player is host in all games
            if player == 0:
                actors = [Actor.remote(get_epsilon(i), learner, buffer, multi_conf,True,config.pretrain,config.portlist[i]) for i in range(num_actors)]
            else:
                actors = [Actor.remote(get_epsilon(i), learner, buffer, "127.0.0.1:"+str(config.portlist[i]),False,config.pretrain) for i in range(num_actors)]
        else:
            actors = [Actor.remote(get_epsilon(i), learner, buffer, multi_conf,False,config.pretrain) for i in range(num_actors)]

        for actor in actors:
            actor.run.remote()
        instance = (buffer,learner,actors)
        instances.append(instance)

        time.sleep(3) #Give the host time to start the game. IMPORTANT

    while not ray.get(buffer.ready.remote()):
        time.sleep(log_interval)
        for player in range(num_players):
            print("Player",player)
            ray.get(instances[player][0].log.remote(log_interval))
            print()

    print('start training')
    for player in range(num_players):
        instances[player][1].run.remote() #starting learners

    done = False
    while not done:
        time.sleep(log_interval)
        for player in range(num_players):
            print("Player",player)
            ray.get(instances[player][0].log.remote(log_interval))
            print()

if __name__ == '__main__':

    train()
