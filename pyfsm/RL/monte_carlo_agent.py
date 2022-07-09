import math
from collections import defaultdict

from gym import Env

from RL.el_agent import ELAgent


class MonteCarloAgent(ELAgent):

    def __init__(self, epsilon=0.1):
        super(MonteCarloAgent, self).__init__(epsilon)

    def learn(self, env: Env, episode_count=1000, gamma=0.9,
              render=False, report_interval=50):
        self.init_log()
        actions = list(range(env.action_space.n))
        self.Q = defaultdict(lambda: [0] * len(actions))
        N = defaultdict(lambda: [0] * len(actions))

        for e in range(episode_count):
            s = env.reset()
            reward = 0.0
            done = False
            experience = []
            while not done:
                if render:
                    env.render()
                a = self.policy(s, actions)
                n_state, reward, done, info = env.step(a)
                experience.append({"state": s, "action": a, "reward": reward})
                s = n_state
            else:
                self.log(reward)

            for i, x in enumerate(experience):
                s, a = x["state"], x["action"]

                G, t = 0, 0
                for j in range(i, len(experience)):
                    G += math.pow(gamma, t) * experience[j]["reward"]
                    t += 1
                # 学習全体お通して行った回数
                N[s][a] += 1
                alpha = 1 / N[s][a]
                self.Q[s][a] += alpha * (G - self.Q[s][a])
                # 上式は以下と同じ
                # self.Q[s][a] = self.Q[s][a] * (1 - alpha) + self.G * alpha

            if e != 0 and e % report_interval == 0:
                self.show_reward_log(episode=e)
