import time
from typing import Optional
import warnings

import gym
import numpy as np
import pygame
import vizdoom.vizdoom as vzd
import random
from filelock import FileLock
import os

# A fixed set of colors for each potential label
# for rendering an image.
# 256 is not nearly enough for all IDs, but we limit
# ourselves here to avoid hogging too much memory.
LABEL_COLORS = np.random.default_rng(42).uniform(25, 256, size=(256, 3)).astype(np.uint8)


class VizdoomEnv(gym.Env):
    def __init__(
        self,
        level,
        frame_skip=1,
        client_args="", #Multiplayer arg is string "IP:Port"
        host = False,
        num_players = 1,
        port = 5060,
        test=False,
        player_name = 'AI',
    ):
        """
        Base class for Gym interface for ViZDoom. Thanks to https://github.com/shakenes/vizdoomgym
        Child classes are defined in vizdoom_env_definitions.py,

        Arguments:
            level (str): path to the config file to load. Most settings should be set by this config file.
            frame_skip (int): how many frames should be advanced per action. 1 = take action on every frame. Default: 1.

        This environment forces window to be hidden. Use `render()` function to see the game.

        Observations are dictionaries with different amount of entries, depending on if depth/label buffers were
        enabled in the config file:
          "rgb"           = the RGB image (always available) in shape (HEIGHT, WIDTH, CHANNELS)
          "depth"         = the depth image in shape (HEIGHT, WIDTH), if enabled by the config file,
          "labels"        = the label image buffer in shape (HEIGHT, WIDTH), if enabled by the config file. For info on labels, access `env.state.labels` variable.
          "automap"       = the automap image buffer in shape (HEIGHT, WIDTH, CHANNELS), if enabled by the config file
          "gamevariables" = all game variables, in the order specified by the config file

        Action space is always a Discrete one, one choice for each button (only one button can be pressed down at a time).
        """
        self.frame_skip = frame_skip
        self.is_multiplayer = len(client_args) > 0 or host

        # init game
        self.level = level
        self.game = vzd.DoomGame()
        self.game.load_config(level)
        self.game.set_window_visible(test) #True for testing purpose
        #self.game.set_window_visible(True)
        #self.lock = FileLock('_vizdoom.ini.lock')

        if test:
            self.game.set_mode(vzd.Mode.ASYNC_PLAYER)
            self.game.set_episode_timeout(0)

        if self.is_multiplayer: #Multiplayer match
            self.game.set_mode(vzd.Mode.ASYNC_PLAYER)
            #safe game variables since ACS skript cant handle specific reward
            if host:
                self.game.add_game_args("-host " + str(num_players) + " "
                # This machine will function as a host for a multiplayer game with this many players (including this machine).
                # It will wait for other machines to connect using the -join parameter and then start the game when everyone is connected.
                "-port {} ".format(port) +  # Specifies the port (default is 5029).
                "+viz_connect_timeout 60 "  # Specifies the time (in seconds), that the host will wait for other players (default is 60).
                "-deathmatch "  # Deathmatch rules are used for the game.
                "+timelimit 10.0 "  # The game (episode) will end after this many minutes have elapsed.
                "+sv_forcerespawn 1 "  # Players will respawn automatically after they die.
                "+sv_noautoaim 1 "  # Autoaim is disabled for all players.
                "+sv_respawnprotect 1 "  # Players will be invulnerable for two second after spawning.
                "+sv_spawnfarthest 1 "  # Players will be spawned as far as possible from any other players.
                "+viz_respawn_delay 10 "  # Sets delay between respawns (in seconds, default is 0).
                "+viz_nocheat 1")  # Disables depth and labels buffer and the ability to use commands that could interfere with multiplayer game.
            else:
                ip, port = client_args.split(':')
                self.game.add_game_args("-join {} -port {}".format(ip,port))  # Connect to a host for a multiplayer game.

            color = random.choice(range(8)) #random player color
            self.game.add_game_args("+name {} +colorset {}".format(player_name,color))


        screen_format = self.game.get_screen_format()
        if screen_format != vzd.ScreenFormat.RGB24:
            warnings.warn(f"Detected screen format {screen_format.name}. Only RGB24 is supported in the Gym wrapper. Forcing RGB24.")
            self.game.set_screen_format(vzd.ScreenFormat.RGB24)

        #with self.lock:
        #    self.game.init()
        self.game.init()

        self.game_variables = [self.game.get_game_variable(vzd.GameVariable.HEALTH),
                               self.game.get_game_variable(vzd.GameVariable.HITCOUNT),
                               self.game.get_game_variable(vzd.GameVariable.SELECTED_WEAPON_AMMO),
                               self.game.get_game_variable(vzd.GameVariable.KILLCOUNT)]

        self.state = None
        self.window_surface = None
        self.isopen = True

        self.depth = self.game.is_depth_buffer_enabled()
        self.labels = self.game.is_labels_buffer_enabled()
        self.automap = self.game.is_automap_buffer_enabled()

        self.num_delta_buttons = 0
        self.all_button_names = []
        for button in self.game.get_available_buttons():
            if "DELTA" in button.name:
                #warnings.warn(f"Removing button {button.name}. DELTA buttons are currently not supported in Gym wrapper. Use binary buttons instead.")
                # Make an 1-setp action for each diretion:
                self.all_button_names.append(button.name+"_POS_"+str(self.num_delta_buttons))
                self.all_button_names.append(button.name+"_NEG_"+str(self.num_delta_buttons))
                self.num_delta_buttons += 1
            else:
                self.all_button_names.append(button.name)

        self.game.set_available_buttons(self.game.get_available_buttons())
        self.action_space = gym.spaces.Discrete(len(self.all_button_names))

        # specify observation space(s)
        self.observation_shape = (self.game.get_screen_height(),self.game.get_screen_width(),3,)
        self.observation_space = gym.spaces.Box(
                                        0,
                                        255,
                                        self.observation_shape,
                                        dtype=np.uint8,
                                    )

    def step(self, action):
        assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call `reset` before using `step` method."
        # convert action to vizdoom action space (one hot)
        act = [0 for _ in range(self.action_space.n - self.num_delta_buttons)]
        if 'DELTA' in self.all_button_names[action]:
            idx_delta = int(self.all_button_names[action].split('_')[-1])
            if self.all_button_names[action].split('_')[-2] == 'NEG':
                act[action-(idx_delta+1)] = -1
            else:
                act[action-idx_delta] = 1
        else:
            act[action] = 1
        reward = self.game.make_action(act, self.frame_skip)

        singelplayer_use_multi_reward = os.path.normpath(self.level).split(os.sep)[-1] == 'multi_single.cfg'
        if self.is_multiplayer or singelplayer_use_multi_reward:
            reward = self.multiplayer_reward()

        self.state = self.game.get_state()
        done = self.game.is_episode_finished()

        #if reward != 0:
        #    print(reward)

        return self.__collect_observations(), reward, done, {} #Only return RGB variant

    #ACS Script reward is global for all players within the map -> Multiplayer reward must be calculated through game variables
    def multiplayer_reward(self):
        reward = 0
        old_health, old_hits, old_ammo, old_frags = self.game_variables
        new_health = self.game.get_game_variable(vzd.GameVariable.HEALTH)
        new_hits = self.game.get_game_variable(vzd.GameVariable.HITCOUNT)
        new_ammo = self.game.get_game_variable(vzd.GameVariable.SELECTED_WEAPON_AMMO)
        new_frags = self.game.get_game_variable(vzd.GameVariable.KILLCOUNT)

        if old_health > new_health and new_health != 0:
            reward -= 20
        elif old_health > new_health and new_health == 0: #player died
            reward -= 100

        if old_ammo > new_ammo:
            reward -= 5

        if old_hits < new_hits:
            reward += 25

        if old_frags < new_frags:
            reward += 100

        self.game_variables = [new_health,new_hits,new_ammo,new_frags]
        return reward

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        if seed is not None:
            self.game.set_seed(seed)
        self.game.new_episode()
        self.state = self.game.get_state()

        if not return_info:
            return self.__collect_observations()
        else:
            return self.__collect_observations(), {}

    def __collect_observations(self):
        if self.state is not None:
            observation = self.state.screen_buffer
        else:
            # there is no state in the terminal step, so a zero observation is returned instead
            observation = np.zeros(self.observation_shape, dtype=np.uint8)

        return observation

    def __build_human_render_image(self):
        """Stack all available buffers into one for human consumption"""
        game_state = self.game.get_state()
        valid_buffers = game_state is not None

        if not valid_buffers:
            # Return a blank image
            num_enabled_buffers = 1 + self.depth + self.labels + self.automap
            img = np.zeros(
                (
                    self.game.get_screen_height(),
                    self.game.get_screen_width() * num_enabled_buffers,
                    3,
                ),
                dtype=np.uint8,
            )
            return img

        image_list = [game_state.screen_buffer]

        if self.depth:
            image_list.append(np.repeat(game_state.depth_buffer[..., None], repeats=3, axis=2))

        if self.labels:
            # Give each label a fixed color.
            # We need to connect each pixel in labels_buffer to the corresponding
            # id via `value``
            labels_rgb = np.zeros_like(game_state.screen_buffer)
            labels_buffer = game_state.labels_buffer
            for label in game_state.labels:
                color = LABEL_COLORS[label.object_id % 256]
                labels_rgb[labels_buffer == label.value] = color
            image_list.append(labels_rgb)

        if self.automap:
            image_list.append(game_state.automap_buffer)

        return np.concatenate(image_list, axis=1)

    def render(self, mode="human"):
        render_image = self.__build_human_render_image()
        if mode == "rgb_array":
            return render_image
        elif mode == "human":
            # Transpose image (pygame wants (width, height, channels), we have (height, width, channels))
            render_image = render_image.transpose(1, 0, 2)
            if self.window_surface is None:
                pygame.init()
                pygame.display.set_caption("ViZDoom")
                self.window_surface = pygame.display.set_mode(render_image.shape[:2])

            surf = pygame.surfarray.make_surface(render_image)
            self.window_surface.blit(surf, (0, 0))
            pygame.display.update()
        else:
            return self.isopen

    def close(self):
        if self.window_surface:
            pygame.quit()
            self.isopen = False
