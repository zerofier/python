import argparse

import gym

from FN.value_function_agent import CartPoleObserver, ValueFunctionTrainer, ValueFunctionAgent


def main(play):
    env = CartPoleObserver(gym.make("CartPole-v1"))
    trainer = ValueFunctionTrainer()
    path = trainer.logger.path_of("value_function_aget.pkl")

    if play:
        agent = ValueFunctionAgent.load(env, path)
        agent.play(env)
    else:
        trained = trainer.train(env)
        trainer.logger.plot("Rewards", trainer.reward_log, trainer.report_interval)
        trained.save(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VF Agent")
    parser.add_argument("--play", action="store_true", help="play with trained model")

    args = parser.parse_args()
    main(args.play)

