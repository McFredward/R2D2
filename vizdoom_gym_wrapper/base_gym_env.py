from typing import Optional
import warnings

import gym
import numpy as np
import pygame
import vizdoom.vizdoom as vzd

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
        test=False,
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

        # init game
        self.game = vzd.DoomGame()
        self.game.load_config(level)
        self.game.set_window_visible(test) #True for testing purpose

        if test:
            self.game.set_mode(vzd.Mode.ASYNC_PLAYER)

        screen_format = self.game.get_screen_format()
        if screen_format != vzd.ScreenFormat.RGB24:
            warnings.warn(f"Detected screen format {screen_format.name}. Only RGB24 is supported in the Gym wrapper. Forcing RGB24.")
            self.game.set_screen_format(vzd.ScreenFormat.RGB24)

        self.game.init()
        self.state = None
        self.window_surface = None
        self.isopen = True

        self.depth = self.game.is_depth_buffer_enabled()
        self.labels = self.game.is_labels_buffer_enabled()
        self.automap = self.game.is_automap_buffer_enabled()

        self.num_delta_buttons = 0
        self.all_button_names = []
        count = 0
        for button in self.game.get_available_buttons():
            if "DELTA" in button.name:
                #warnings.warn(f"Removing button {button.name}. DELTA buttons are currently not supported in Gym wrapper. Use binary buttons instead.")
                # Make an 1-setp action for each diretion:
                self.num_delta_buttons += 1
                self.all_button_names.append(button.name+"_POS_"+str(count))
                count += 1
                self.all_button_names.append(button.name+"_NEG_"+str(count))
                count += 1
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
            offset = int(self.all_button_names[action].split('_')[-1])
            if self.all_button_names[action].split('_')[-2] == 'NEG':
                act[action-offset] = -1
            else:
                act[action-offset] = 1
        else:
            act[action] = 1

        reward = self.game.make_action(act, self.frame_skip)
        self.state = self.game.get_state()
        done = self.game.is_episode_finished()

        return self.__collect_observations(), reward, done, {} #Only return RGB variant

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
