from typing import Type

import numpy as np
from gym import Env

from RL.el_agent import ELAgent


class Actor(ELAgent):
    def __init__(self, env: Env):
        super().__init__(epsilon=-1)
        nrow = env.observation_space.n
        ncol = env.action_space.n
        self.actions = list(range(ncol))
        self.Q = np.random.uniform(0, 1, nrow * ncol).reshape((nrow, ncol))

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def policy(self, s, random=True):
        if random:
            a = np.random.choice(self.actions, 1, p=self.softmax(self.Q[s]))
        else:
            a = [np.argmax(self.Q[s])]
        return a[0]


class Critic:
    def __init__(self, env: Env):
        states = env.observation_space.n
        self.V = np.zeros(states)


class ActorCritic:
    def __init__(self, actor_class: Type, critic_class: Type):
        self.actor_class = actor_class
        self.critic_class = critic_class

    def train(self, env: Env, episode_count=1000, gamma=0.9,
              learning_rate=0.1, render=False, report_interval=50):
        actor = self.actor_class(env)
        critic = self.critic_class(env)

        actor.init_log()
        for e in range(episode_count):
            s = env.reset()
            reward = 0.0
            done = False
            while not done:
                if render:
                    env.render()

                a = actor.policy(s)
                n_state, reward, done, info = env.step(a)

                gain = reward + gamma * critic.V[n_state]
                estimated = critic.V[s]
                td = gain - estimated
                actor.Q[s][a] += learning_rate * td
                critic.V[s] += learning_rate * td
                s = n_state
            else:
                actor.log(reward)

            if e != 0 and e % report_interval == 0:
                actor.show_reward_log(episode=e)

        return actor, critic

