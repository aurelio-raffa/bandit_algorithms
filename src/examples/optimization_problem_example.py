from src.optimization.__dependencies import *
from src.optimization.budget_optimizer import budget_optimizer


if __name__ == '__main__':
    mat_1 = np.array([
        [0, 0, 0, 0, 0, 0],
        [3, 4, 6, 7, 8, 2],
        [3, 5, 2, 4, 5, 2],
        [3, 5, 6, 2, 5, 2]])
    mat_2 = np.array([
        [0, 0, 0, 0],
        [4, 5, 6, 1],
        [3, 6, 1, 2],
        [3, 2, 5, 2]])
    budget_values = [0, 1, 2, 3]

    budget_optimizer([mat_1, mat_2], budget_values, pedantic=True)
