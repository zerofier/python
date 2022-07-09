from collections import defaultdict

from gym import Env

from RL.el_agent import ELAgent


class QLearningAgent(ELAgent):
    def __init__(self, epsilon=0.1):
        super(QLearningAgent, self).__init__(epsilon)

    def learn(self, env: Env, episode_count=1000, gamma=0.9,
              learning_rate=0.1, render=False, report_interval=50):
        self.init_log()
        actions = list(range(env.action_space.n))
        self.Q = defaultdict(lambda: [0] * len(actions))

        for e in range(episode_count):
            s = env.reset()
            reward = 0.0
            done = False
            while not done:
                if render:
                    env.render()

                a = self.policy(s, actions)
                n_state, reward, done, info = env.step(a)

                gain = reward + gamma * max(self.Q[n_state])
                estimated = self.Q[s][a]
                self.Q[s][a] += learning_rate * (gain - estimated)
                s = n_state
            else:
                self.log(reward)

            if e != 0 and e % report_interval == 0:
                self.show_reward_log(episode=e)
