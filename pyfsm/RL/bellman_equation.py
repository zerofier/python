
STATE_1 = 'happy_end'
STATE_2 = 'bad_end'

ACTION_1 = 'up'
ACTION_2 = 'down'


def V(s, gamma=0.99):
    return R(s) + gamma * max_V_on_next_state(s)


def R(s):
    if s == STATE_1:
        return 1
    elif s == STATE_2:
        return -1
    else:
        return 0


def max_V_on_next_state(s):
    if s in [STATE_1, STATE_2]:
        return 0

    actions = [ACTION_1, ACTION_2]
    values = []
    for a in actions:
        transition_prods = transit_func(s, a)
        v = 0
        for next_state in transition_prods:
            prob = transition_prods[next_state]
            v += prob * V(next_state)
        values.append(v)

    return max(values)


LIMIT_GATE_COUNT = 5
HAPPY_END_BORDER = 4
MOVE_PROB = 0.9


def transit_func(s, a):
    actions = s.split('_')[1:]

    def next_state(_state, _action):
        return '_'.join([_state, _action])

    if len(actions) == LIMIT_GATE_COUNT:
        up_count = sum(1 if a == ACTION_1 else 0 for a in actions)
        state = STATE_1 if up_count >= HAPPY_END_BORDER else STATE_2
        prob = 1.0
        return {state: prob}
    else:
        opposite = ACTION_1 if a == ACTION_2 else ACTION_2
        return {
            next_state(s, a): MOVE_PROB,
            next_state(s, opposite): 1 - MOVE_PROB
        }
