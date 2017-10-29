EPSILON = 0.15  # for get_eigmin


def get_eigmin(W):  # flake8
    # eig_vals = np.linalg.eigvals(W.reshape(2, 2))
    # return np.amin(eig_vals)
    print(W)
    # M = DET(A) / (TRACE(A) + e)
    a, b, c, d = W
    return (a * d - b * c) / (a + d + EPSILON)


def get_all_eigmin(W_xx, W_xy, W_yy):
    return (W_xx * W_yy - W_xy ** 2) / (W_xx + W_yy + EPSILON)
