import numpy as np

from RL.action import Action
from RL.state import State


class Environment:
    def __init__(self, grid, move_prod=0.8):
        self.grid = grid
        self.agent_state = State()

        self.default_reword = -0.04

        self.move_prod = move_prod
        self.reset()

    @property
    def row_length(self):
        return len(self.grid)

    @property
    def col_length(self):
        return len(self.grid[0])

    @property
    def actions(self):
        return [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]

    @property
    def states(self):
        states = []
        for row in range(self.row_length):
            for col in range(self.col_length):
                if self.grid[row][col] != 9:
                    states.append(State(row, col))
        return states

    def transit_func(self, state: State, action: Action):
        transition_prods = {}
        if not self.can_action_at(state):
            return transition_prods

        opposite_direction = Action(action.value * -1)

        for a in self.actions:
            prob = 0
            if a == action:
                prob = self.move_prod
            elif a != opposite_direction:
                prob = (1 - self.move_prod) / 2

            next_state = self._move(state, a)
            if next_state not in transition_prods:
                transition_prods[next_state] = prob
            else:
                transition_prods[next_state] += prob

        return transition_prods

    def can_action_at(self, state: State):
        if self.grid[state.row][state.col] == 0:
            return True
        else:
            return False

    def _move(self, state: State, action: Action):
        if not self.can_action_at(state):
            raise Exception("Can't move from here!")

        next_state = state.clone()

        if action == Action.UP:
            next_state.row -= 1
        elif action == Action.DOWN:
            next_state.row += 1
        elif action == Action.LEFT:
            next_state.col -= 1
        elif action == Action.RIGHT:
            next_state.col += 1

        if not (0 <= next_state.row < self.row_length):
            next_state = state
        if not (0 <= next_state.col < self.col_length):
            next_state = state

        if self.grid[next_state.row][next_state.col] == 9:
            next_state = state

        return next_state

    def reward_func(self, state: State):
        reward = self.default_reword
        done = False

        attr = self.grid[state.row][state.col]
        if attr == 1:
            reward = 1
            done = True
        elif attr == -1:
            reward = -1
            done = True

        return reward, done

    def reset(self):
        self.agent_state = State(self.row_length - 1, 0)
        return self.agent_state

    def step(self, action: Action):
        next_state, reward, done = self.transit(self.agent_state, action)
        if next_state is not None:
            self.agent_state = next_state

        return next_state, reward, done

    def transit(self, state: State, action: Action):
        transition_prods = self.transit_func(state, action)
        if len(transition_prods) == 0:
            return None, None, True

        next_states = []
        prods = []
        for s in transition_prods:
            next_states.append(s)
            prods.append(transition_prods[s])

        next_state = np.random.choice(next_states, p=prods)
        reward, done = self.reward_func(next_state)
        return next_state, reward, done
