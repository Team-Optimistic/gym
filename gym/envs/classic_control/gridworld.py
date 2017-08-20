"""
Deterministic, fully observable Grid World environment. Taken from Sutton & Barto 1996
source: http://people.inf.elte.hu/lorincz/Files/RL_2006/SuttonBook.pdf
"""

import logging

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

logger = logging.getLogger(__name__)


class GridWorld(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    N = 0
    S = 1
    E = 2
    W = 3

    @staticmethod
    def string_to_action(action):
        if action == 'N':
            return GridWorld.N
        elif action == 'S':
            return GridWorld.S
        elif action == 'E':
            return GridWorld.E
        elif action == 'W':
            return GridWorld.W

    @staticmethod
    def action_to_string(action):
        if action == GridWorld.N:
            return 'N'
        elif action == GridWorld.S:
            return 'S'
        elif action == GridWorld.E:
            return 'E'
        elif action == GridWorld.W:
            return 'W'

    def __init__(self):
        # move N,S,E,W
        self.action_space = spaces.Discrete(4)
        # row, column (y, x)
        self.observation_space = spaces.MultiDiscrete([[0, 4], [0, 4]])
        self._seed()
        self.state = np.array([0, 0])
        self.viewer = None

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action), "Invalid action"

        y = self.state[0]
        x = self.state[1]
        # compute new coordinates and rewards
        if y == 0 and x == 1:
            x = 1
            y = 4
            reward = 10
        elif y == 0 and x == 3:
            x = 3
            y = 2
            reward = 5
        elif (action == GridWorld.N and y == 0) or \
                (action == GridWorld.S and y == 4) or \
                (action == GridWorld.E and x == 4) or \
                (action == GridWorld.W and x == 0):
            reward = -1
        elif action == GridWorld.N:
            y -= 1
            reward = 0
        elif action == GridWorld.S:
            y += 1
            reward = 0
        elif action == GridWorld.E:
            x += 1
            reward = 0
        elif action == GridWorld.W:
            x -= 1
            reward = 0
        else:
            reward = 0

        self.state[0] = y
        self.state[1] = x
        return self.state, reward, False, {}

    def _reset(self):
        self.state[0] = self.np_random.randint(0, 5)
        self.state[1] = self.np_random.randint(0, 5)
        return self.state

    def _render(self, mode='human', close=False, values=None):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        size = 5
        px_per_cell = 30
        size_px = size * px_per_cell

        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(size_px, size_px)
            for x in range(size + 1):
                for y in range(size + 1):
                    x_line = rendering.Line((px_per_cell * x, 0), (px_per_cell * x, size_px))
                    y_line = rendering.Line((0, px_per_cell * y), (size_px, px_per_cell * y))
                    self.viewer.add_geom(x_line)
                    self.viewer.add_geom(y_line)

            self.robot = self.viewer.draw_circle(radius=px_per_cell / 3, filled=True)
            self.viewer.add_geom(self.robot)
            self.robot_transform = rendering.Transform()
            self.robot.add_attr(self.robot_transform)

        self.robot_transform.set_translation(self.state[1] * px_per_cell + px_per_cell / 2,
                                             size_px - self.state[0] * px_per_cell - px_per_cell / 2)

        return self.viewer.render(return_rgb_array=(mode == 'rgb_array'))
