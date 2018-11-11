import numpy as np
from scipy.linalg import solve


class Helper:

    def __init__(self) -> None:
        A = np.array([[12.0, 7.0, 3.0], [1.0, 5.0, 1.0], [2.0, 7.0, -11.0]])
        x1 = np.array([[1], [1], [1]])
        b = np.array([[2], [-5], [6]])
        self.gauss_seidel(A, b, x1, 1000)

        print(solve(A, b))  # solve linear equation system

    def gauss_seidel(self, A, b, x, n):
        D = np.diag(np.diag(A))
        L = np.tril(A, -1)
        U = np.triu(A, 1)

        P = D + L
        P1 = np.linalg.inv(P)

        iterations = 0
        while iterations < n:
            iterations += 1
            # np.dot(A, B) => A * B
            x = np.dot(P1, (np.dot(-U, x) + b))  # xk+1 = (D + L)^-1 (-U * xk + b)

        print(x)


if __name__ == '__main__':
    Helper()
