import numpy as np

if __name__ == "__main__":

    qs = np.matrix([[1, 2, 0], [3, 4, 1], [2, -2, 0]], dtype=float)

    print qs.shape
    print type(qs)

    (m, n) = qs.shape
    Q = np.zeros((m, n))

    for k in range(n):
        Q[:, [k]] = qs[:, k] / np.linalg.norm(qs[:, k])

        for j in range(k + 1, n):
            qk = Q[:, k]
            aj = qs[:, j]
            r = np.dot(qk, aj)[0, 0]
            qs[:, [j]] -= (qk * r).reshape((m, 1))

    print np.dot(Q[:, 0], Q[:, 1])
    print np.dot(Q[:, 1], Q[:, 2])
    print np.dot(Q[:, 0], Q[:, 2])