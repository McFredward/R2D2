from typing import Tuple
import numpy as np
import numba as nb
import config

#Binary SUM TREE -> Saves the priority in the leafes. Parent contain sum of two leafs
def create_ptree(capacity: int) -> Tuple[int, np.ndarray]:
    num_layers = 1
    while capacity > 2**(num_layers-1):
        num_layers += 1

    ptree = np.zeros(2**num_layers-1, dtype=np.float64)
    return num_layers, ptree

@nb.jit(nopython=True, cache=True)
def ptree_update(num_layers: int, ptree: np.ndarray, prio_exponent: float, td_error: np.ndarray, idxes: np.ndarray):
    priorities = td_error ** prio_exponent

    #Formula to convert array indices to leaf indices of the tree
    idxes = idxes + 2**(num_layers-1) - 1
    ptree[idxes] = priorities #Save in the leafs of the tree

    #Build the sum tree around the leaf-values
    for _ in range(num_layers-1):
        idxes = (idxes-1) // 2 #parent index
        idxes = np.unique(idxes) #delete double entrys (two leafs have the same parent)
        ptree[idxes] = ptree[2*idxes+1] + ptree[2*idxes+2] #sum up the children

def ptree_sample(num_layers: int, ptree: np.ndarray, is_exponent: float, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    if config.use_prioritized_replay:
        return ptree_sample_prioritized(num_layers,ptree,is_exponent,num_samples)
    else:
        return ptree_sample_uniformly(num_layers,num_samples)


@nb.jit(nopython=True, cache=True)
def ptree_sample_prioritized(num_layers: int, ptree: np.ndarray, is_exponent: float, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    p_sum = ptree[0] # whole sum
    interval = p_sum / num_samples

    prefixsums = np.arange(0, p_sum, interval, dtype=np.float64) + np.random.uniform(0, interval, num_samples)

    idxes = np.zeros(num_samples, dtype=np.int64)
    for _ in range(num_layers-1):
        nodes = ptree[idxes*2+1]
        idxes = np.where(prefixsums < nodes, idxes*2+1, idxes*2+2) #go the path with the highest sum
        prefixsums = np.where(idxes%2 == 0, prefixsums - ptree[idxes-1], prefixsums)

    # importance sampling weights
    priorities = ptree[idxes]
    min_p = np.min(priorities)
    is_weights = np.power(priorities/min_p, -is_exponent)

    idxes -= 2**(num_layers-1) - 1

    return idxes, is_weights

@nb.jit(nopython=True, cache=True)
def ptree_sample_uniformly(num_layers: int, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    #get all leafs
    idxes = np.array(range(0,2**(num_layers-1) - 1))
    idxes = np.random.choice(idxes,size=num_samples)
    is_weights = np.ones(idxes.shape)

    return idxes, is_weights
