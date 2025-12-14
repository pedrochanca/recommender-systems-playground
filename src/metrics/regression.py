from typing import List, Tuple
import math


def calc_squared_error(user_targets: List[Tuple[float, float]]) -> Tuple[float, int]:
    """
    Calculates sum of squared errors and count for a user.
    """
    if not user_targets:
        return 0.0, 0

    sq_error = sum((pred - true) ** 2 for pred, true in user_targets)
    return sq_error, len(user_targets)


def calc_rmse(user_targets: List[Tuple[float, float]]) -> float:
    """
    Calculates RMSE for a single user (or global list).
    """

    sq_error, n_users = calc_squared_error(user_targets)
    return math.sqrt(sq_error / n_users)
