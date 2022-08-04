'''open.ai gym environment wrapper.
reference: https://github.com/openai/gym/tree/master/gym/wrappers'''
from collections import deque
import gym
import numpy as np
import cv2
import config
import vizdoom_gym_wrapper

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """start the game with no-op actions to provide random starting positions
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1) #pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        return self.env.step(action)

class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """cilp reward in range [-1, 1]."""
        return np.clip(reward, -1, 1)


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84):
        """
        Warp frames to 84x84 as done in the Nature paper and later work.
        """
        super().__init__(env)
        self._width = width
        self._height = height

        original_space = self.observation_space
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width),
            dtype=np.uint8,
        )
        #print("TEST",original_space.dtype, len(original_space))
        #assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    def observation(self, frame):

        #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        frame = cv2.resize(
            frame, (self._width, self._height), interpolation=cv2.INTER_AREA
        )

        return frame


def create_env(env_name=config.game_name+config.env_type, clip_rewards=True,multi_conf="",is_host =False,testing=False,port=5060,num_players=config.num_players,name='AI'):

    if config.game_name == "CartPole":
        env = gym.make('CartPole' + config.env_type)

        env = WarpFrame(env)
        #env = NoopResetEnv(env)

    else:
        env = gym.make(env_name,frame_skip=config.frame_skip,client_args=multi_conf,host=is_host,num_players=num_players,test=testing,port=port,player_name=name)

        env = WarpFrame(env)

        if clip_rewards:
            env = ClipRewardEnv(env)
        #if noop_start:
        #    env = NoopResetEnv(env)

    return env
