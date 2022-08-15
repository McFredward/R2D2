import random
import time
import torch
import numpy as np
import ray
from worker import Learner, Actor, ReplayBuffer
import config
import logging

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

    NUM_AGENTS = 2
    TOP_LIMIT = 1
    GENERATIONS = 10

    agents = []

    multi_conf = ""

    # Initial setup. Used for the base agent.
    start_config = {"batch size": config.batch_size, "prio_exp": config.prio_exponent, "prio_bias": config.importance_sampling_exponent,
                    "lr": config.lr, "dueling": config.use_dueling, "epsilon": config.base_eps, "shape": config.obs_shape,
                    "frame skip": config.frame_skip, "gamma": config.gamma, "burn in": config.burn_in_steps}

    # First agent is the same as start config
    agents.append(create_agent_from_config(start_config, multi_conf, num_actors))

    for ii in range(NUM_AGENTS - 1):
        agents.append(mutate(start_config))

    elite_index = None

    # -- Init done -- start training
    logger = logging.getLogger('genetic_logs')

    for g in range(GENERATIONS):
        rewards = run_agents_n_times(agents, 2, num_actors, log_interval)

        sorted_parent_indexes = np.argsort(rewards)[::-1][:TOP_LIMIT]

        top_rewards = []
        for best_parent in sorted_parent_indexes:
            top_rewards.append(rewards[best_parent])

        logger.info("Generation ", g, " | Mean rewards: ", np.mean(rewards), " | Mean of top 3: ", np.mean(top_rewards[:3]))
        logger.info("Top ", TOP_LIMIT, " scores", sorted_parent_indexes)
        logger.info("Rewards for top: ", top_rewards)

        children, elite_index = generate_children([arg[-1] for arg in agents], sorted_parent_indexes, elite_index)
        agents = children


def create_agent_from_config(conf: dict, multi_conf="", num_actors=config.num_actors):

    buffer = ReplayBuffer.remote(batch_size=conf["batch size"], alpha=conf["prio_exp"], beta=conf["prio_bias"])
    learner = Learner.remote(buffer=buffer, lr=conf["lr"], use_dueling=conf["dueling"])

    if config.multiplayer:
        base_host_actor = Actor.remote(get_epsilon(0, conf["epsilon"]), learner, buffer, multi_conf, True, config.pretrain)
        actors = [base_host_actor] + [Actor.remote(get_epsilon(i, conf["epsilon"]), learner, buffer,
                                                        "127.0.0.1:5029", False, config.pretrain, obs_shape=conf["shape"],
                                                        frame_skip=conf["frame skip"], gamma=conf["gamma"],
                                                        buffer_burn_in_steps=conf["burn in"],
                                                        use_dueling=conf["dueling"]) for i in range(1,num_actors)]
    else:
        actors = [Actor.remote(get_epsilon(i, conf["epsilon"]), learner, buffer, multi_conf, False,
                                    config.pretrain, obs_shape=conf["shape"], frame_skip=conf["frame skip"], gamma=conf["gamma"],
                                    buffer_burn_in_steps=conf["burn in"], use_dueling=conf["dueling"]) for i in range(num_actors)]

    return [buffer, learner, actors, conf]


def generate_children(agent_confs, parent_indexes, elite_index):
    # Generate children from the best performing agents

    children_agents = []

    #first take selected parents from sorted_parent_indexes and generate N-1 children
    for i in range(len(agent_confs)-1):
        selected_agent_index = parent_indexes[np.random.randint(len(parent_indexes))]
        children_agents.append(mutate(agent_confs[selected_agent_index]))

    #now add one elite
    elite_child = add_elite(agent_confs, parent_indexes, elite_index)
    children_agents.append(elite_child)
    elite_index=len(children_agents)-1 #it is the last one

    return children_agents, elite_index


def add_elite(agent_confs, sorted_parent_indexes, elite_index=None, only_consider_top_n=10):
    # Select the elite of agents (best performing)

    candidate_elite_index = sorted_parent_indexes[:only_consider_top_n]

    if(elite_index is not None):
        candidate_elite_index = np.append(candidate_elite_index,[elite_index])

    top_score = None
    top_elite_index = None

    for i in candidate_elite_index:
        score = avg_score(create_agent_from_config(agent_confs[i]), n=5)
        print("Score for elite cadidiate i ", i, " is ", score)

        if(top_score is None):
            top_score = score
            top_elite_index = i
        elif(score > top_score):
            top_score = score
            top_elite_index = i

    print("Elite selected with index ", top_elite_index, " and score", top_score)

    child_agent = create_agent_from_config(agent_confs[top_elite_index])
    return child_agent


def run_agents_n_times(agents, n, num_actors, log_interval):
    """
    Run the agents one by one

    :param agents:
    :param n:
    :param num_actors:
    :param log_interval:
    :return:
    """

    rewards_agents = []

    for agent in agents:
        rewards_agents.append(avg_score(agent, n, log_interval))

    return rewards_agents


def avg_score(agent, n, log_interval):
    """
    Run agent

    :param agent:
    :param n:
    :param log_interval:
    :return:
    """

    buffer = agent[0]
    learner = agent[1]
    actors = agent[2]
    conf = agent[3]

    for actor in actors:
        actor.run.remote()

    while not ray.get(buffer.ready.remote()):
        time.sleep(log_interval)
        ray.get(buffer.log.remote(log_interval))
        print()

    learner.run.remote()

    done = False
    while not done:
        time.sleep(log_interval)
        done = ray.get(buffer.log.remote(log_interval))
        print()

    # Needs implementation
    reward = ray.get(learner.get_reward.remote())

    return reward


def mutate(conf, mutation_power=0.02):
    """
    Mutate the config and create a new Agent based on the mutated config

    :param conf: The old config (the mutation will be performed on)
    :param mutation_power: the intensity of mutation for every value
    :return: an agent consisting of [buffer, learner, actors, conf]
    """

    no_conf_vals = len(conf)
    keys = conf.keys()
    values = conf.values()

    conf_vals_to_mutate = np.random.choice([0, 1], size=no_conf_vals, p=[.5, .5])
    new_conf = {}
    for ii in range(no_conf_vals):
        if conf_vals_to_mutate[ii]:
            new_conf[keys[ii]] = mutate_value(values[ii], mutation_power)

    return create_agent_from_config(new_conf)



def mutate_value(old_value, mutation_power):
    """
    Mutate a single value

    :param old_value: Old value to mutate
    :param mutation_power: Intensity of mutation (for numbers)
    :return: new value (same type as old value)
    """

    if type(old_value) == bool:
        new_value = not(old_value)
    elif type(old_value) == int:
        new_value = np.round(old_value + np.random.normal * mutation_power)
    elif type(old_value) == float:
        new_value = old_value + np.random.normal * mutation_power
    elif type(old_value) == tuple or type(old_value) == list:
        new_value = (mutate_value(old_value[ii]) for ii in range(len(old_value)))
    else:
        raise TypeError("Unknown type for mutation")

    return new_value


if __name__ == '__main__':

    train()

