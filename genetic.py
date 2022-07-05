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

    NUM_AGENTS = 1
    TOP_LIMIT = 1
    GENERATIONS = 50

    agents = []

    multi_conf = ""
    for ii in range(NUM_AGENTS):
        buffer = ReplayBuffer.remote()
        learner = Learner.remote(buffer)

        if config.multiplayer:
            host_actor = Actor.remote(get_epsilon(0), learner, buffer, multi_conf, True,config.pretrain)
            actors = [host_actor] + [Actor.remote(get_epsilon(i), learner, buffer, "127.0.0.1:5029",False,config.pretrain) for i in range(1,num_actors)]
        else:
            actors = [Actor.remote(get_epsilon(i), learner, buffer, multi_conf, False, config.pretrain) for i in range(num_actors)]

        agents.append([buffer, learner, actors])

    elite_index = None

    for g in range(GENERATIONS):
        rewards = run_agents_n_times(agents, 2, num_actors, log_interval)

        sorted_parent_indexes = np.argsort(rewards)[::-1][:TOP_LIMIT]

        print("")

        top_rewards = []
        for best_parent in sorted_parent_indexes:
            top_rewards.append(rewards[best_parent])

        print("Generation ", g, " | Mean rewards: ", np.mean(rewards), " | Mean of top 3: ", np.mean(top_rewards[:3]))
        #print(rewards)
        print("Top ", TOP_LIMIT, " scores", sorted_parent_indexes)
        print("Rewards for top: ", top_rewards)

        print("")

        children = generate_children(agents, sorted_parent_indexes, elite_index)
        agents = children


def generate_children(agents, parent_indexes, elite_index):
    # Generate children from the best performing agents

    children_agents = []

    #first take selected parents from sorted_parent_indexes and generate N-1 children
    for i in range(len(agents)-1):
        selected_agent_index = parent_indexes[np.random.randint(len(parent_indexes))]
        children_agents.append(mutate(agents[selected_agent_index]))

    #now add one elite
    elite_child = add_elite(agents, parent_indexes, elite_index)
    children_agents.append(elite_child)
    elite_index=len(children_agents)-1 #it is the last one

    return children_agents, elite_index


def add_elite(agents, sorted_parent_indexes, elite_index=None, only_consider_top_n=10):
    # Select the elite of agents (best performing)

    candidate_elite_index = sorted_parent_indexes[:only_consider_top_n]

    if(elite_index is not None):
        candidate_elite_index = np.append(candidate_elite_index,[elite_index])

    top_score = None
    top_elite_index = None

    for i in candidate_elite_index:
        score = avg_score(agents[i], n=5)
        print("Score for elite i ", i, " is ", score)

        if(top_score is None):
            top_score = score
            top_elite_index = i
        elif(score > top_score):
            top_score = score
            top_elite_index = i

    print("Elite selected with index ", top_elite_index, " and score", top_score)

    child_agent = copy.deepcopy(agents[top_elite_index])
    return child_agent


def run_agents_n_times(agents, n, num_actors, log_interval):
    # Run the agents (after each other)

    rewards_agents = []

    for agent in agents:
        avg_score(agent, n, log_interval)

    return rewards_agents


def avg_score(agent, n, log_interval):
    # Run a single agent

    buffer = agent[0]
    learner = agent[1]
    actors = agent[2]

    for actor in actors:
        actor.run.remote()

    while not ray.get(buffer.ready.remote()):
        time.sleep(log_interval)
        ray.get(buffer.log.remote(log_interval))
        print()

    print('start training')
    learner.run.remote()

    done = False
    while not done:
        time.sleep(log_interval)
        done = ray.get(buffer.log.remote(log_interval))
        print()

    reward = ray.get(learner.get_reward.remote())

    return reward


def mutate(agent, mutation_power=0.02):
    # Mutation

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

