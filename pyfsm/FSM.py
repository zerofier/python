from abc import ABC, abstractmethod
from typing import Generic, Type, TypeVar


EntityType = TypeVar('EntityType')


class BaseState(ABC, Generic[EntityType]):
    """
    the base of state class
    """
    @abstractmethod
    def enter(self, entity: EntityType):
        """
        called at state changing.
        :param entity:
        :return:
        """
        pass

    @abstractmethod
    def execute(self, entity: EntityType):
        """
        called at state updating
        :param entity:
        :return:
        """
        pass

    @abstractmethod
    def exit(self, entity: EntityType):
        """
        called at state changing.
        :param entity:
        :return:
        """
        pass


class StateMachine(Generic[EntityType]):
    """

    """
    def __init__(self, owner: EntityType, init_state: Type[BaseState[EntityType]]):
        self.owner: EntityType = owner
        self.current_state: Type[BaseState[EntityType]] = None
        self.previous_state: Type[BaseState[EntityType]] = None
        self.global_state: Type[BaseState[EntityType]] = None
        self.change_state(init_state)

    def set_current_state(self, state: Type[BaseState[EntityType]]):
        self.current_state = state

    def set_previous_state(self, state: Type[BaseState[EntityType]]):
        self.previous_state = state

    def set_global_state(self, state: Type[BaseState[EntityType]]):
        self.global_state = state

    def update(self):
        """
        execute current state
        :return:
        """
        if self.global_state:
            self.global_state.execute(self.owner)

        if self.current_state:
            self.current_state.execute(self.owner)

    def change_state(self, new_state: Type[BaseState[EntityType]]):
        """
        change to new state
        :param new_state:
        :return:
        """
        self.previous_state = self.current_state

        if self.current_state:
            self.current_state.exit(self.owner)

        self.current_state = new_state

        self.current_state.enter(self.owner)

    def revert_to_previous(self):
        self.change_state(self.previous_state)


class BaseEntity:
    pass
