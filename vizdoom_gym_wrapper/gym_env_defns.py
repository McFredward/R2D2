import os
from .base_gym_env import VizdoomEnv
from vizdoom import scenarios_path


class VizdoomScenarioEnv(VizdoomEnv):
    """Basic ViZDoom environments which reside in the `scenarios` directory"""
    def __init__(
        self, scenario_file, frame_skip=1, client_args="", host = False,num_players = 1, test=False
    ):
        super(VizdoomScenarioEnv, self).__init__(
           os.path.join(scenarios_path, scenario_file), frame_skip, client_args, host,num_players, test
        )
