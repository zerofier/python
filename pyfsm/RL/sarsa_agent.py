from gym import Env

from RL.el_agent import ELAgent
from collections import defaultdict


class SARSAAgent(ELAgent):
    def __init__(self, epsilon=0.1):
        super(SARSAAgent, self).__init__(epsilon)

    def learn(self, env: Env, episode_count=1000, gamma=0.9,
              learning_rate=0.1, render=False, report_interval=50):

        self.init_log()
        actions = list(range(env.action_space.n))
        self.Q = defaultdict(lambda: [0] * len(actions))
        for e in range(episode_count):
            s = env.reset()
            reward = 0.0
            done = False
            a = self.policy(s, actions)
            while not done:
                if render:
                    env.render()
                n_state, reward, done, info = env.step(a)

                n_action = self.policy(n_state, actions)
                gain = reward + gamma * self.Q[n_state][n_action]
                estimated = self.Q[s][a]
                self.Q[s][a] += learning_rate * (gain - estimated)
                s = n_state
                a = n_action
            else:
                self.log(reward)

            if e != 0 and e % report_interval == 0:
                self.show_reward_log(episode=e)
