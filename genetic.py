import random
import time
import torch
import numpy as np
import ray
from worker import Learner, Actor, ReplayBuffer
import config
import logging
import os

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.set_num_threads(1)
torch.set_grad_enabled(False)

@ray.remote
class is_running_struct:
    def __init__(self):
        self.is_running = True
    def terminate(self):
        self.is_running = False
    def get_is_running(self):
        return self.is_running


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

    if not os.path.exists('logfiles'):
        os.mkdir('logfiles')
    else:
        logdir_files = os.listdir(os.path.join(os.getcwd(),'logfiles'))
        if(len(logdir_files) != 0):
            print("Deleting all existing log files! ARE YOU SURE?")
            print("Press any key to continue..")
            input()
            print(logdir_files)
            for file in logdir_files:
                os.remove(os.path.join(os.getcwd(),'logfiles',file))

    # Initial setup. Used for the base agent.
    start_config = {"prio_exp": config.prio_exponent, "prio_bias": config.importance_sampling_exponent, "batch size": int(config.batch_size),
                    "lr": config.lr, "dueling": config.use_dueling,"double": config.use_double , "epsilon": config.base_eps,
                    "frame skip": int(config.frame_skip), "gamma": config.gamma, "burn in": int(config.burn_in_steps), "player_idx": 0}

    # First agent is the same as start config
    agents.append(create_agent_from_config(start_config, 0, multi_conf, num_actors))

    for ii in range(NUM_AGENTS - 1):
        start_config_copy = start_config.copy()
        start_config_copy['player_idx'] = ii+1
        agents.append(mutate(start_config_copy,0))

    elite_index = None
    # -- Init done -- start training
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('genetic_logs')
    logger.addHandler(logging.FileHandler(os.path.join('logfiles','genetic.log'), 'w'))

    for g in range(GENERATIONS):
        logger.info("--- Start Generation No {} ---".format(g))
        for agent in agents: #print current configs in log
            logger.info("Agent {}:\n{}".format(agent[3]['player_idx'],str(agent[3])))

        rewards = run_master(agents, 2, num_actors, log_interval)

        logger.info("Generation No. {} finished!".format(g))

        # Top agents (sorted descending)
        sorted_parent_indexes = np.argsort(rewards)[::-1][:TOP_LIMIT]

        top_rewards = []
        for best_parent in sorted_parent_indexes:
            top_rewards.append(rewards[best_parent])
        #logger.info just takes one string: cant use it like print(.,.,.,.,.)
        logger.info("Generation "+str(g)+" | Mean rewards: "+str(np.mean(rewards))+" | Mean of top 3: "+str(np.mean(top_rewards[:3])))
        logger.info("Top "+str(TOP_LIMIT)+" scores "+str(sorted_parent_indexes))
        logger.info("Rewards for top: "+str(top_rewards))

        children, elite_index = generate_children([arg[3] for arg in agents], sorted_parent_indexes,g+1)
        agents = children


def create_agent_from_config(conf: dict, generation_idx, multi_conf="", num_actors=config.num_actors):

    r_buffer = ReplayBuffer.remote(conf["player_idx"], generation_idx, batch_size=conf["batch size"], alpha=conf["prio_exp"], beta=conf["prio_bias"])
    learner = Learner.remote(conf["player_idx"], generation_idx, buffer=r_buffer, lr=conf["lr"], use_dueling=conf["dueling"], use_double=conf["double"])

    actors_is_running = [is_running_struct.remote() for _ in range(num_actors)]

    if config.multiplayer:
        base_host_actor = Actor.remote(actors_is_running[0], get_epsilon(0, conf["epsilon"]), learner, r_buffer, multi_conf, True, config.pretrain,
                                       use_dueling=conf["dueling"], use_double=conf["double"], frame_skip=conf["frame skip"], gamma=conf["gamma"],
                                       buffer_burn_in_steps=conf["burn in"])
        actors = [base_host_actor] + [Actor.remote(actors_is_running[i], get_epsilon(i, conf["epsilon"]), learner, r_buffer,
                                                   "127.0.0.1:5029", False, config.pretrain,
                                                   frame_skip=conf["frame skip"], gamma=conf["gamma"],
                                                   buffer_burn_in_steps=conf["burn in"],
                                                   use_dueling=conf["dueling"]) for i in range(1,num_actors)]
    else:
        actors = [Actor.remote(actors_is_running[i], get_epsilon(i, conf["epsilon"]), learner, r_buffer, multi_conf, False,
                               config.pretrain, use_dueling=conf["dueling"], use_double=conf["double"], frame_skip=conf["frame skip"],
                               gamma=conf["gamma"], buffer_burn_in_steps=conf["burn in"],) for i in range(num_actors)]

    return [r_buffer, learner, actors, conf, actors_is_running]


def generate_children(agent_confs: list[dict], parent_indexes: list, generation_idx):
    # Generate children from the best performing agents

    children_agents = []

    # Fill not-top agents with random mutations of top performing agents
    for i in range(len(agent_confs)-1):
        # Take a random index of a parent
        selected_agent_index = parent_indexes[np.random.randint(len(parent_indexes))]
        # Mutate the config
        agent_conf = agent_confs[selected_agent_index].copy()
        agent_conf['player_idx'] = i+1
        children_agents.append(mutate(agent_conf,generation_idx))

    # Add one elite
    # elite_child = add_elite(agent_confs, parent_indexes, elite_index)
    # children_agents.append(elite_child)
    # elite_index=len(children_agents)-1 #it is the last one

    # Add best of top to children
    agent_conf = agent_confs[parent_indexes[0]].copy()
    agent_conf['player_idx'] = 0
    children_agents.append(create_agent_from_config(agent_conf,generation_idx))
    elite_index = parent_indexes[0]

    return children_agents, elite_index

"""
def add_elite(agent_confs: list[dict], sorted_parent_indexes: list, elite_index=None, only_consider_top_n: int=1):
    # Select the elite of agents (best performing)

    # Best of the best
    candidate_elite_index = sorted_parent_indexes[:only_consider_top_n]

    # Previous elite is also considered
    if elite_index is not None:
        candidate_elite_index = np.append(candidate_elite_index, [elite_index])

    top_score = None
    top_elite_index = None

    # Compare new elite(s) and previous elite
    for i in candidate_elite_index:
        agent_confs[i]["player_idx"] = int(agent_confs[i]["player_idx"] * 10 + i)
        score = run_agent(create_agent_from_config(agent_confs[i]), n=5)
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
"""

def run_master(agents, n: int, num_actors: int, log_interval): #agents: list[list[ReplayBuffer, Learner, list[Actor], dict, is_running:_struct]]
    """
    Start all masterthreads of all agents, and waits for their completion

    :param agents:
    :param n:
    :param num_actors:
    :param log_interval:
    :return:
    """

    rewards_agents = []
    #Start all agent master-threads
    agents_is_running_struct_lst = [is_running_struct.remote() for _ in range(len(agents))]
    for agent_idx in range(len(agents)):
        run_agent.remote(agents_is_running_struct_lst[agent_idx],agents[agent_idx], n, log_interval)

    #Wait for all agents to finish
    still_running = True
    while still_running:
        time.sleep(log_interval)
        still_running = False
        for agent_idx in range(len(agents)):
            # Only keeps beeing False if all "is_runnning" from all actors in False
            is_running = ray.get(agents_is_running_struct_lst[agent_idx].get_is_running.remote())
            #print("still_running = " + str(still_running) + " or " + str(is_running) + " = " + str(still_running or is_running))
            still_running = still_running or is_running


    #Grab the sum reward of all agents
    for agent in agents:
        # Buffer stores the reward
        r_buffer = agent[0]
        rewards_agents.append(ray.get(r_buffer.get_reward.remote()))

    return rewards_agents

@ray.remote(num_cpus=1)
def run_agent(agent_is_running_struct,agent, n: int, log_interval): #list[ReplayBuffer, Learner, list[Actor], dict, is_running_struct]
    """
    Agent master-thread. Starting all actors, the buffer and the learner and then waits for its completion

    :param global_is_running_struct:
    :param agent:
    :param n:
    :param log_interval:
    :return: None
    """

    r_buffer = agent[0]
    learner = agent[1]
    actors = agent[2]
    #conf = agent[3]
    is_running_struct = agent[4]

    for actor in actors:
        actor.run.remote()

    while not ray.get(r_buffer.ready.remote()):
        time.sleep(log_interval)
        ray.get(r_buffer.log.remote(log_interval))
        print()

    learner.run.remote()

    done = False
    while not done: #Wait for training steps reach maximum training steps
        time.sleep(log_interval)
        r_buffer.log.remote(log_interval) #print to logfile
        current_training_steps = ray.get(r_buffer.get_num_training_steps.remote())
        done = current_training_steps >= config.training_steps
        print()

    print("----->DONE<-------")
    for actor_idx in range(len(actors)):
        ray.get(is_running_struct[actor_idx].terminate.remote())

    ray.get(agent_is_running_struct.terminate.remote())


def mutate(conf: dict, generation_idx, mutation_power=0.002):
    """
    Mutate the config and create a new Agent based on the mutated config

    :param conf: The old config (the mutation will be performed on)
    :param mutation_power: the intensity of mutation for every value
    :return: an agent consisting of [buffer, learner, actors, conf]
    """

    no_conf_vals = len(conf)
    keys = list(conf.keys())
    values = list(conf.values())

    print("conf",conf)

    conf_vals_to_mutate = np.random.choice([0, 1], size=no_conf_vals-1, p=[.5, .5])
    new_conf = {}
    for ii in range(no_conf_vals):
        if ii == no_conf_vals-1: # Last entry is player_idx
            new_conf[keys[ii]] = int(conf[keys[ii]])
        elif conf_vals_to_mutate[ii]:
            new_conf[keys[ii]] = mutate_value(values[ii], mutation_power)
        else:
            new_conf[keys[ii]] = values[ii]

    return create_agent_from_config(new_conf,generation_idx)



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
        new_value = abs(int(np.round(old_value + np.random.normal(0, 1) * mutation_power)))
    elif type(old_value) == float:
        new_value = abs(old_value + np.random.normal(0, 1) * mutation_power)
    elif type(old_value) == tuple or type(old_value) == list:
        new_value = (mutate_value(old_value[ii]) for ii in range(len(old_value)))
    else:
        raise TypeError("Unknown type for mutation")

    return new_value


if __name__ == '__main__':

    train()

