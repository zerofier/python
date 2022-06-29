import random

from RL.env import Environment
from RL.state import State


class Agent:
    def __init__(self, env: Environment):
        self.actions = env.actions

    def policy(self, state: State):
        return random.choice(self.actions)
