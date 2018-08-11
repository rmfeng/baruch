"""
Approximating Implied Volatilites from SSRN-id2908494.pdf
"""
import numpy as np


def calc_abc(y, R):
    inner = np.exp((1 - 2/np.pi) * y) - np.exp(-(1 - 2/np.pi) * y)
    innerB = np.exp((1 - 2 / np.pi) * y) + np.exp(-(1 - 2 / np.pi) * y)
    A = inner ** 2

    B_t1 = 4 * (np.exp(2 * y/np.pi) + np.exp(-2 * y/np.pi))
    B_t2 = 2 * np.exp(-y) * innerB * (np.exp(2 * y) + 1 - R ** 2)
    B = B_t1 - B_t2

    C = np.exp(-2 * y) * (R ** 2 - (np.exp(y) - 1) ** 2) * ((np.exp(y) + 1) ** 2 - R ** 2)
    return A, B, C


def impvol_approx_call(px, K, T, S, r, q):
    F = S * np.exp((r - q) * T)
    # print("F", F)
    disc = np.exp(-r * T)
    # print("disc", disc)
    y = np.log(F/K)
    # print("y", y)
    alpha = px / (K * disc)
    # print("alpha", alpha)
    R = 2 * alpha - np.exp(y) + 1
    # print("R", R)
    A, B, C = calc_abc(y, R)
    # print("A=%s, B=%s, C=%s" % (A, B, C))

    beta = (2 * C) / (B + np.sqrt(B ** 2 + (4 * A * C)))
    # print("beta", beta)
    gamma = -np.pi * np.log(beta) / 2
    # print("gamma", gamma)

    if y >= 0:
        C0 = K * disc * (np.exp(y) * ploya_A(np.sqrt(2 * y)) - 0.5)
        # print("C0", C0)
        if px <= C0:
            # print("y>=0, px <= C0")
            return (1 / np.sqrt(T)) * (np.sqrt(gamma + y) - np.sqrt(gamma - y))
        else:
            # print("y>=0, px < C0")
            return (1 / np.sqrt(T)) * (np.sqrt(gamma + y) + np.sqrt(gamma - y))
    else:
        C0 = K * disc * ((np.exp(y) / 2) - ploya_A(-np.sqrt(-2 * y)))
        # print("C0", C0)
        if px <= C0:
            # print("y<0, px <= C0")
            return (1 / np.sqrt(T)) * (-np.sqrt(gamma + y) + np.sqrt(gamma - y))
        else:
            # print("y<0, px > C0")
            return (1 / np.sqrt(T)) * (np.sqrt(gamma + y) + np.sqrt(gamma - y))


def impvol_approx_put(px, K, T, S, r, q):
    F = S * np.exp((r - q) * T)
    # print("F", F)
    disc = np.exp(-r * T)
    # print("disc", disc)
    y = np.log(F/K)
    # print("y", y)
    alpha = px / (K * disc)
    # print("alpha", alpha)
    R = 2 * alpha + np.exp(y) - 1
    # print("R", R)
    A, B, C = calc_abc(y, R)
    # print("A=%s, B=%s, C=%s" % (A, B, C))

    beta = (2 * C) / (B + np.sqrt(B ** 2 + (4 * A * C)))
    # print("beta", beta)
    gamma = -np.pi * np.log(beta) / 2
    # print("gamma", gamma)

    if y >= 0:
        P0 = K * disc * (0.5 - (np.exp(y) * ploya_A(-np.sqrt(2 * y))))
        # print("C0", C0)
        if px <= P0:
            # print("y>=0, px <= C0")
            return (1 / np.sqrt(T)) * (np.sqrt(gamma + y) - np.sqrt(gamma - y))
        else:
            # print("y>=0, px < C0")
            return (1 / np.sqrt(T)) * (np.sqrt(gamma + y) + np.sqrt(gamma - y))
    else:
        P0 = K * disc * (ploya_A(np.sqrt(-2 * y)) - np.exp(y) / 2)
        # print("C0", C0)
        if px <= P0:
            # print("y<0, px <= C0")
            return (1 / np.sqrt(T)) * (-np.sqrt(gamma + y) + np.sqrt(gamma - y))
        else:
            # print("y<0, px > C0")
            return (1 / np.sqrt(T)) * (np.sqrt(gamma + y) + np.sqrt(gamma - y))


def ploya_A(x):
    if x >= 0:
        return 0.5 + 0.5 * np.sqrt(1 - np.exp(-2 * (x ** 2) / np.pi))
    else:
        return 0.5 - 0.5 * np.sqrt(1 - np.exp(-2 * (x ** 2) / np.pi))
