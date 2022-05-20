import random
from typing import Type

import FSM


class Miner(FSM.BaseEntity):
    """
    the miner class
    """

    def __init__(self, init_state: Type[FSM.BaseState]):
        self.state_machine = FSM.StateMachine[Miner](self, init_state)
        self.life = 100
        self.stamina = 100

    def update(self):
        self.state_machine.update()

    def change_state(self, new_state):
        self.state_machine.change_state(new_state)

    def play(self):
        self.stamina -= int(random.normalvariate(30, 10))
        print(f"M: {self.stamina}, L: {self.life}")
        if self.stamina <= 30:
            self.state_machine.change_state(SleepState.instance())

    def sleep(self):
        self.stamina = 100
        self.life -= 1
        if self.life <= 0:
            self.state_machine.change_state(DeadState.instance())
        else:
            self.state_machine.change_state(PlayState.instance())

    def is_life(self):
        return self.life > 0


class PlayState(FSM.BaseState[Miner]):
    """
    the play state of miner
    """

    def __init__(self):
        pass

    def enter(self, entity: Miner):
        print("play state enter")

    def execute(self, entity: Miner):
        print("play state execute")
        entity.play()

    def exit(self, entity: Miner):
        print("play state exit")

    _instance = None

    @classmethod
    def instance(cls):
        if not cls._instance:
            cls._instance = cls()
        return cls._instance


class SleepState(FSM.BaseState[Miner]):
    """
    the sleep state of miner
    """

    def __init__(self):
        pass

    def enter(self, entity: Miner):
        print("sleep state enter")

    def execute(self, entity: Miner):
        print("sleep state execute")
        entity.sleep()

    def exit(self, entity: Miner):
        print("sleep state exit")

    _instance = None

    @classmethod
    def instance(cls):
        if not cls._instance:
            cls._instance = cls()
        return cls._instance


class DeadState(FSM.BaseState[Miner]):
    def __init__(self):
        pass

    def enter(self, entity: Miner):
        print("dead state enter")

    def execute(self, entity: Miner):
        print("dead state execute")

    def exit(self, entity: Miner):
        print("dead state exit")

    _instance = None

    @classmethod
    def instance(cls):
        if not cls._instance:
            cls._instance = cls()
        return cls._instance
