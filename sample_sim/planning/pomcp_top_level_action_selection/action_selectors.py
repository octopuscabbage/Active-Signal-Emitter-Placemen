import enum


class ActionSelectors(enum.Enum):
    UCB = "ucb",
    UGapEb  = "ugapeb"


def discrete_integer_to_arm_selection(x):
    if x is None:
        return None
    elif x == 1:
        return ActionSelectors.UCB
    elif x == 2:
        return ActionSelectors.UGapEb
    else:
        raise Exception(f"Don't understand action selector {x}")

