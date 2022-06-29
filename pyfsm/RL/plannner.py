from typing import Dict

from RL.action import Action
from RL.env import Environment
from RL.state import State


class Planner:
    def __init__(self, env: Environment):
        self.env = env
        self.log = []

    def initialize(self):
        self.env.reset()

    def plan(self, gamma=0.9, threshold=1e-4):
        raise Exception("Planner have to implements plan method.")

    def transition_at(self, state: State, action: Action):
        transition_prods = self.env.transit_func(state, action)
        for next_state in transition_prods:
            prod = transition_prods[next_state]
            reward, _ = self.env.reward_func(next_state)
            yield prod, next_state, reward

    def dict_to_grid(self, state_reward_dict: Dict[State, float]):
        grid = []
        for i in range(self.env.row_length):
            row = [0.0] * self.env.col_length
            grid.append(row)

        for s in state_reward_dict:
            grid[s.row][s.col] = state_reward_dict[s]

        return grid


class ValueIterationPlanner(Planner):
    def __init__(self, env: Environment):
        super().__init__(env)

    def plan(self, gamma=0.9, threshold=1e-4):
        self.initialize()
        actions = self.env.actions
        V = {}
        for s in self.env.states:
            V[s] = 0.0

        while True:
            delta = 0.0
            self.log.append(self.dict_to_grid(V))
            for s in V:
                if not self.env.can_action_at(s):
                    continue

                expected_rewards = []
                for a in actions:
                    r = 0.0
                    for prod, next_state, reward in self.transition_at(s, a):
                        r += prod * (reward + gamma + V[next_state])
                    expected_rewards.append(r)

                max_reward = max(expected_rewards)
                delta = max(delta, abs(max_reward - V[s]))
                V[s] = max_reward

            if delta < threshold:
                break

        V_grid = self.dict_to_grid(V)
        return V_grid
