# Ideas from https://github.com/pmocz/nbody-python/blob/master/nbody.py

import numpy as np

def get_accel(R, M):
    
    softening_parameter = 0.0001
    """
    Compute the gravitational accelerations of masses.
    R is an N x 3 matrix of space positions, units of [cm].
    M is a length N vector of masses, units of [g].
    Returns an N x 3 matrix of accelerations, units of [cm/s^2].
    """
    # get N x 1 matrices for position
    X = R[:, 0:1]
    Y = R[:, 1:2]
    Z = R[:, 2:3]

    # compute deltas (N x N) (all pairwise particle separations: r_j - r_i)
    DX = X.T - X
    DY = Y.T - Y
    DZ = Z.T - Z

    # compute 1/R^3 for each pair
    IR3 = (DX**2 + DY**2 + DZ**2 + softening_parameter**2)**(-1.5)

    # gravitational constant
    G = 6.67259e-8 # [cm^3/g/s^2]
    
    # accelerations

    AX = (DX*IR3)@M 
    AY = (DY*IR3)@M
    AZ = (DZ*IR3)@M
    A  = G * np.array((AX, AY, AZ)).T

    return A