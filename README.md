# Changes compared to the original repo
The focus of the fork is to implement DOOM Agents using the R2D2 reinforcement model.
It contains (mainly) the following changes and additions:

* Adapted Vizdoom gym-wrapper to work with the model
* Added DELTA-Buttons to the gym-wrapper
* Added hyperparameter to turn off Double-DQN extension
* Added hyperparameter to turn off Dueling-DQN extension
* It is now possible to set `prio_exponent` to zero (to turn off prioritized replay)
* Added multiplayer training (multiple R2D2 models playing against each other; 1 actor from each model in `num_actors` parallel games)
* Adapted `test.py` to replay learned Vizdoom sessions (Single or Multiplayer)
* Added `plot.py` which visualizes the generated log-files.
* Added genetic algorithm (`genetic` branch) for hyperparameter-search.
* Several small fixes and changes

# R2D2 (Recurrent Experience Replay in Distributed Reinforcement Learning)
## introduction
An Implementation of [Recurrent Experience Replay in Distributed Reinforcement Learning](https://openreview.net/forum?id=r1lyTjAqYX) (Kapturowski et al. 2019) in PyTorch and Ray.

## Training
First adjust parameter settings in config.py (number of actors, environment name, etc.).

Then run:
```
python3 train.py
```
## Testing
```
python3 test.py
```
## Result
The code was trained and tested in Atari game 'Boxing'
 ![image](https://github.com/ZiyuanMa/R2D2/blob/main/images/Boxing.jpg)






