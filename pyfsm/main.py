from RL.agent import Agent
# from RL.bellman_equation import V
from RL.env import Environment

# from miner import Miner, PlayState
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
    print(plan)

    planner = PolicyIterationPlanner(env)
    plan = planner.plan()
    print(plan)


if __name__ == '__main__':
    main()
    # print(V('state'))
    # print(V('state_up_up'))
    # print(V('state_down_down'))
