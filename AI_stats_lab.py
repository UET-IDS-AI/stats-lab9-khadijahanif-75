import numpy as np


# -------------------------------------------------
# Sparse 4 by 4 Joint PMF
# -------------------------------------------------

PMF_TABLE = {
    (0, 0): 0.10, (0, 1): 0.05, (0, 2): 0.00, (0, 3): 0.00,
    (1, 0): 0.15, (1, 1): 0.20, (1, 2): 0.05, (1, 3): 0.00,
    (2, 0): 0.00, (2, 1): 0.10, (2, 2): 0.15, (2, 3): 0.05,
    (3, 0): 0.00, (3, 1): 0.00, (3, 2): 0.05, (3, 3): 0.10
}


def joint_pmf(x, y):
    """
    Joint PMF table.
    """
    return PMF_TABLE.get((x, y), 0)


def marginal_px(x):
    """
    Compute PX(x) by summing over y.
    """
    return sum(joint_pmf(x, y) for y in range(4))


def marginal_py(y):
    """
    Compute PY(y) by summing over x.
    """
    return sum(joint_pmf(x, y) for x in range(4))


def conditional_pmf_x_given_y(x, y):
    """
    Compute P(X=x given Y=y).
    """
    py = marginal_py(y)

    if py == 0:
        return 0

    return joint_pmf(x, y) / py


def conditional_distribution_x_given_y(y):
    """
    Conditional distribution dictionary.
    """
    return {
        x: conditional_pmf_x_given_y(x, y)
        for x in range(4)
    }


def probability_sum_greater_than_3():
    """
    Compute P(X + Y > 3).
    """
    total = 0

    for x in range(4):
        for y in range(4):
            if x + y > 3:
                total += joint_pmf(x, y)

    return total


def independence_check():
    """
    Check if X and Y are independent.
    """
    for x in range(4):
        for y in range(4):

            lhs = joint_pmf(x, y)
            rhs = marginal_px(x) * marginal_py(y)

            if not np.isclose(lhs, rhs):
                return False

    return True


# -------------------------------------------------
# Expectation, Covariance, and Correlation
# -------------------------------------------------

def expected_x():
    """
    Compute E[X].
    """
    return sum(x * marginal_px(x) for x in range(4))


def expected_y():
    """
    Compute E[Y].
    """
    return sum(y * marginal_py(y) for y in range(4))


def expected_xy():
    """
    Compute E[XY].
    """
    total = 0

    for x in range(4):
        for y in range(4):
            total += x * y * joint_pmf(x, y)

    return total


def variance_x():
    """
    Compute Var(X).
    """
    ex = expected_x()

    ex2 = sum((x ** 2) * marginal_px(x) for x in range(4))

    return ex2 - (ex ** 2)


def variance_y():
    """
    Compute Var(Y).
    """
    ey = expected_y()

    ey2 = sum((y ** 2) * marginal_py(y) for y in range(4))

    return ey2 - (ey ** 2)


def covariance_xy():
    """
    Compute Cov(X,Y).
    """
    return expected_xy() - expected_x() * expected_y()


def correlation_xy():
    """
    Compute correlation coefficient.
    """
    cov = covariance_xy()
    varx = variance_x()
    vary = variance_y()

    return cov / np.sqrt(varx * vary)


def variance_sum():
    """
    Compute Var(X+Y).
    """
    exy_sum = 0
    exy_sum_sq = 0

    for x in range(4):
        for y in range(4):

            p = joint_pmf(x, y)

            exy_sum += (x + y) * p
            exy_sum_sq += ((x + y) ** 2) * p

    return exy_sum_sq - (exy_sum ** 2)


def variance_identity_check():
    """
    Verify:

    Var(X+Y) = Var(X) + Var(Y) + 2*Cov(X,Y)

    Return True if the identity holds, else False.
    """

    lhs = variance_sum()

    rhs = (
        variance_x()
        + variance_y()
        + 2 * covariance_xy()
    )

    return bool(np.isclose(lhs, rhs))
# -------------------------------------------------
# Testing
# -------------------------------------------------

if __name__ == "__main__":

    print("P(X+Y > 3):", probability_sum_greater_than_3())

    print("Independent:", independence_check())

    print("E[X]:", expected_x())
    print("E[Y]:", expected_y())

    print("E[XY]:", expected_xy())

    print("Var(X):", variance_x())
    print("Var(Y):", variance_y())

    print("Cov(X,Y):", covariance_xy())

    print("Correlation:", correlation_xy())

    print("Var(X+Y):", variance_sum())

    print("Variance Identity Holds:",
          variance_identity_check())
