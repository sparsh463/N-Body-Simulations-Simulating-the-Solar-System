from numba import njit
import numpy as np

@njit(parallel=True, cache=True)
def get_accel(R, M):
    """
    Compute the gravitational accelerations of masses.
    R is an N x 3 matrix of space positions, units of [cm].
    M is a length N vector of masses, units of [g].
    Returns an N x 3 matrix of accelerations, units of [cm/s^2].
    
    """
    N = R.shape[0]
    A = np.zeros_like(R)
    G = 6.67259e-8  # [cm^3/g/s^2]
    eps = 1e-3  # Softening factor to prevent division by zero

    for i in range(N):
        for j in range(N):
            if i != j:
                dx = R[j, 0] - R[i, 0]
                dy = R[j, 1] - R[i, 1]
                dz = R[j, 2] - R[i, 2]
                inv_r3 = ((dx**2 + dy**2 + dz**2 + eps**2)**(-1.5))
                A[i, 0] += M[j] * dx * inv_r3
                A[i, 1] += M[j] * dy * inv_r3
                A[i, 2] += M[j] * dz * inv_r3

    A *= G
    return A
