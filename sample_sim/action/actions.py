import numpy as np
import enum

class ActionXYT(enum.Enum):
    STAY_STILL = 0
    POSITIVE_X = 1
    NEGATIVE_X = 2
    POSITIVE_Y = 3
    NEGATIVE_Y = 4


class ActionXYZ(enum.Enum):
    POSITIVE_X = 0
    NEGATIVE_X = 1
    POSITIVE_Y = 2
    NEGATIVE_Y = 3
    POSITIVE_Z = 4
    NEGATIVE_Z = 5

    #STAY_STILL = 6

class ActionXY(enum.Enum):
    POSITIVE_X = 0
    NEGATIVE_X = 1
    POSITIVE_Y = 2
    NEGATIVE_Y = 3


class ActionModel(enum.Enum):
    XYT = 0
    XYZ = 1
    XY = 2


def action_enum(model: ActionModel):
    if model == ActionModel.XYT:
        return ActionXYT
    elif model == ActionModel.XYZ:
        return ActionXYZ
    elif model == ActionModel.XY:
        return ActionXY
    else:
        raise Exception(f"action model not understood {model}")


def apply_action_to_state(state, action, step_sizes):
    if isinstance(action, ActionXYT):
        state = np.array(state)
        if action == ActionXYT.POSITIVE_X:
            sprime = state + np.array([step_sizes[0], 0, step_sizes[2]])
        elif action == ActionXYT.NEGATIVE_X:
            sprime = state + np.array([-1 * step_sizes[0], 0, step_sizes[2]])
        elif action == ActionXYT.POSITIVE_Y:
            sprime = state + np.array([0, step_sizes[1], step_sizes[2]])
        elif action == ActionXYT.NEGATIVE_Y:
            sprime = state + np.array([0, -1 * step_sizes[1], step_sizes[2]])
        elif action == ActionXYT.STAY_STILL:
            sprime = state + np.array([0, 0, step_sizes[2]])
        else:
            raise Exception(f"Action not understood: {action}")
    elif isinstance(action, ActionXY):
        state = np.array(state)
        if action == ActionXY.POSITIVE_X:
            sprime = state + np.array([step_sizes[0], 0])
        elif action == ActionXY.NEGATIVE_X:
            sprime = state + np.array([-1 * step_sizes[0], 0])
        elif action == ActionXY.POSITIVE_Y:
            sprime = state + np.array([0, step_sizes[1]])
        elif action == ActionXY.NEGATIVE_Y:
            sprime = state + np.array([0, -1 * step_sizes[1]])
        else:
            raise Exception(f"Action not understood: {action}")

    elif isinstance(action, ActionXYZ):
        state = np.array(state)
        if action == ActionXYZ.POSITIVE_X:
            sprime = state + np.array([step_sizes[0], 0, 0])
        elif action == ActionXYZ.NEGATIVE_X:
            sprime = state + np.array([-1 * step_sizes[0], 0, 0])
        elif action == ActionXYZ.POSITIVE_Y:
            sprime = state + np.array([0, step_sizes[1], 0])
        elif action == ActionXYZ.NEGATIVE_Y:
            sprime = state + np.array([0, -1 * step_sizes[1], 0])
        elif action == ActionXYZ.POSITIVE_Z:
            sprime = state + np.array([0, 0, step_sizes[2]])
        elif action == ActionXYZ.NEGATIVE_Z:
            sprime = state + np.array([0, 0, -1 * step_sizes[2]])
        # elif action == ActionXYZ.STAY_STILL:
        #     sprime = state
        else:
            raise Exception(f"Action not understood: {action}")

    else:
        raise Exception(f"action_model not understood: {action_model}")

    return sprime
