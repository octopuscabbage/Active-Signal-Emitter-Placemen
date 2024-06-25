from math import floor, log10

import numpy as np



def is_good_matrix(x):
    """
    >>> is_good_matrix(np.array([float('NaN'),0.0]))
    False
    >>> is_good_matrix(np.array([float('inf'),0.0]))
    False
    >>> is_good_matrix(np.array([float('-inf'),0.0]))
    False
    >>> is_good_matrix(np.random.random((100,100)))
    True

    Returns true if all the elements of x are finite and not NaN
    :param x: test matrix
    :return:
    """
    s = np.sum(x)
    return (not np.isnan(s)) and np.isfinite(s)


def stacked_meshgrid(*args):
    t = np.meshgrid(*args)
    t = tuple(map(lambda x: x.flatten(),t))
    t_X = np.stack(t, axis=-1)
    return t_X

def compute_weights(a):
    out = np.abs(a) / np.sum(np.abs(a))
    assert np.isclose(np.sum(out),1)
    return out

def rwmse(gt,y):
    return np.sqrt(np.sum(compute_weights(gt) * np.square(gt - y)))
def wmae(gt,y):
    return np.sum(compute_weights(gt) * np.abs(gt - y))


#     X_t_out = np.zeros(X_t.shape)
#     X_t_out[:,0] = (X_t[:,0] - workspace.xmin) / (workspace.xmax - workspace.xmin)
#     X_t_out[:,1] = (X_t[:,1] - workspace.ymin) / (workspace.ymax - workspace.ymin)
#     X_t_out[:,2] = (X_t[:,2] - workspace.zmin) / (workspace.zmax - workspace.zmin)
#     return X_t_out

def unit_vector_between_two_points(pi,pj):
    """
    >>> unit_vector_between_two_points(np.array([0,0,0]),np.array([1,0,0]))
    array([1., 0., 0.])
    >>> np.isclose(unit_vector_between_two_points(np.array([5,5,5]),np.array([7,7,7])),unit_vector_between_two_points(np.array([5,5,5]),np.array([8,8,8]))).all()
    True

    :param pi:
    :param pj:
    :return:
    """
    #Unit vector pointing from pi to pj
    #https://math.stackexchange.com/questions/12745/how-do-you-calculate-the-unit-vector-between-two-points
    diff = pj - pi
    return diff / np.linalg.norm(diff,ord=2)

def round_sigfigs(x, sig):
    return '{:g}'.format(float('{:.{p}g}'.format(x, p=sig)))
