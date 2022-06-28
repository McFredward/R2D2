import copy
import random
import time
import torch
import numpy as np
import ray
from worker import Learner, Actor, ReplayBuffer
import config

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.set_num_threads(1)
torch.set_grad_enabled(False)

def get_epsilon(actor_id: int, base_eps: float = config.base_eps, alpha: float = config.alpha, num_actors: int = config.num_actors):
    exponent = 1 + actor_id / (num_actors-1) * alpha
    return base_eps**exponent


def train(num_actors=config.num_actors, log_interval=config.log_interval):
    ray.init()

    NUM_AGENTS = 3
    TOP_LIMIT = 1
    GENERATIONS = 100

    agents = []

    for ii in range(NUM_AGENTS):
        buffer = ReplayBuffer.remote()
        learner = Learner.remote(buffer)
        actors = [Actor.remote(get_epsilon(i), learner, buffer) for i in range(num_actors)]

        agents.append([buffer, learner, actors])

    child_agent = mutate(agents[0])

    # for actor in actors:
    #     actor.run.remote()
    #
    # while not ray.get(buffer.ready.remote()):
    #     time.sleep(log_interval)
    #     ray.get(buffer.log.remote(log_interval))
    #     print()
    #
    # print('start training')
    # learner.run.remote()
    #
    # done = False
    # while not done:
    #     time.sleep(log_interval)
    #     done = ray.get(buffer.log.remote(log_interval))
    #     print()

def mutate(agent, mutation_power=0.02):

    agent_mutated = copy.deepcopy(agent)
    actors = agent_mutated[2]
    for actor in actors:
        actor.update_params.remote(mutation_power)

    learner = agent_mutated[1]
    learner.update_params.remote(mutation_power)

    buffer = agent_mutated[0]

    print("Test", flush=True)
    print(ray.get(learner.get_weights.remote()))

    return [buffer, learner, actors]

if __name__ == '__main__':

    train()

