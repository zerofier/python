from pprint import pprint

import gym
import numpy as np

from RL.actor_critic import ActorCritic, Actor, Critic
from RL.agent import Agent, EpsilonGreedyAgent
# from RL.bellman_equation import V
from RL.cointoss import CoinToss
from RL.env import Environment
# from miner import Miner, PlayState
from RL.frozen_lake_util import show_q_value
from RL.plannner import ValueIterationPlanner, PolicyIterationPlanner


def main():
    # bob = Miner(PlayState.instance())
    #
    # while bob.is_life():
    # 	bob.update()

    grid = [[0, 0, 0, 1],
            [0, 9, 0, -1],
            [0, 0, 0, 0]]

    env = Environment(grid)
    agent = Agent(env)

    for i in range(10):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.policy(state)
            next_state, reward, done = env.step(action)
            total_reward += reward
            state = next_state

        print(f"Episode {i}: Agent gets {total_reward} reward.")

    planner = ValueIterationPlanner(env)
    plan = planner.plan()
    print("ValueIterationPlanner:")
    pprint(plan, width=120)

    planner = PolicyIterationPlanner(env)
    plan = planner.plan()
    print("PolicyIterationPlanner:")
    pprint(plan, width=120)


def train():
    # agent = MonteCarloAgent(epsilon=0.1)
    # agent = QLearningAgent(epsilon=0.1)
    # agent = SARSAAgent(epsilon=0.1)
    trainer = ActorCritic(Actor, Critic)
    # env = gym.make("FrozenLakeEasy-v0")
    env = gym.make("Taxi-v3")
    actor, critic = trainer.train(env, episode_count=3000)
    # agent.learn(env, episode_count=500)
    # show_q_value(agent.Q)
    # agent.show_reward_log()
    # show_q_value(actor.Q, env)
    # actor.show_reward_log()

    for _ in range(5):
        s = env.reset()
        done = False
        while not done:
            env.render()
            a = actor.policy(s)
            n_state, reward, done, info = env.step(a)
            s = n_state


if __name__ == '__main__':
    import pandas as pd
    import matplotlib.pyplot as plt

    # main()

    # print(V('state'))
    # print(V('state_up_up'))
    # print(V('state_down_down'))
    def main():
        env = CoinToss([0.1, 0.5, 0.1, 0.9, 0.1])
        epsilons = [0.0, 0.1, 0.2, 0.5, 0.8]
        game_steps = list(range(10, 310, 10))
        result = {}
        for e in epsilons:
            agent = EpsilonGreedyAgent(epsilon=e)
            means = []
            for s in game_steps:
                env.max_episode_steps = s
                rewards = agent.play(env)
                means.append(np.mean(rewards))
            result[f"epsilon={e}"] = means
        result["coin toss count"] = game_steps
        result = pd.DataFrame(result)
        result.set_index("coin toss count", drop=True, inplace=True)
        result.plot.line(figsize=(10, 5))
        plt.show()

    # main()
    train()
