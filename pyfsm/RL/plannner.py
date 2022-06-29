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

    def plan(self, gamma: float = 0.9, threshold: float = 1e-4):
        raise Exception("Planner have to implements plan method.")

    def transition_at(self, state: State, action: Action):
        transition_prods = self.env.transit_func(state, action)
        for next_state in transition_prods:
            prob = transition_prods[next_state]
            reward, _ = self.env.reward_func(next_state)
            yield prob, next_state, reward

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

    def plan(self, gamma: float = 0.9, threshold: float = 1e-4):
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
                    for prob, next_state, reward in self.transition_at(s, a):
                        r += prob * (reward + gamma * V[next_state])
                    expected_rewards.append(r)

                max_reward = max(expected_rewards)
                delta = max(delta, abs(max_reward - V[s]))
                V[s] = max_reward

            if delta < threshold:
                break

        V_grid = self.dict_to_grid(V)
        return V_grid


class PolicyIterationPlanner(Planner):
    def __init__(self, env: Environment):
        super().__init__(env)
        self.policy = {}

    def initialize(self):
        super().initialize()
        self.policy = {}
        actions = self.env.actions
        states = self.env.states
        for s in states:
            self.policy[s] = {}
            for a in actions:
                self.policy[s][a] = 1 / len(actions)

    def estimate_by_policy(self, gamma: float, threshold: float):
        V = {}
        for s in self.env.states:
            V[s] = 0.0

        while True:
            delta = 0.0
            for s in V:
                expected_reward = []
                for a in self.policy[s]:
                    action_prob = self.policy[s][a]
                    r = 0.0
                    for prob, next_state, reward in self.transition_at(s, a):
                        r += action_prob * prob * (reward + gamma * V[next_state])
                    expected_reward.append(r)
                value = sum(expected_reward)
                delta = max(delta, abs(value - V[s]))
                V[s] = value

            if delta < threshold:
                break

        return V

    def plan(self, gamma: float = 0.9, threshold: float = 1e-4):
        self.initialize()
        states = self.env.states
        actions = self.env.actions

        def take_max_action(action_value_dict):
            return max(action_value_dict, key=action_value_dict.get)

        while True:
            update_stable = True
            V = self.estimate_by_policy(gamma, threshold)
            self.log.append(self.dict_to_grid(V))

            for s in states:
                policy_action = take_max_action(self.policy[s])

                action_rewards = {}
                for a in actions:
                    r = 0.0
                    for prob, next_state, reward in self.transition_at(s, a):
                        r += prob * (reward + gamma * V[next_state])
                    action_rewards[a] = r

                best_action = take_max_action(action_rewards)
                if policy_action != best_action:
                    update_stable = False

                for a in self.policy[s]:
                    prob = 1 if a == best_action else 0
                    self.policy[s][a] = prob

            if update_stable:
                break

        V_grid = self.dict_to_grid(V)
        return V_grid
