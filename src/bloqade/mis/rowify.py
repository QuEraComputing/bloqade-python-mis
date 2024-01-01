import numpy as np
from decimal import Decimal


def round(x, resolution):

    def do_round(x: Decimal) -> Decimal:
        return int(x / resolution) * resolution

    str_iter = map(str, x)
    dec_iter = map(Decimal, str_iter)
    rounded_iter = map(do_round, dec_iter)
    float_iter = map(float, rounded_iter)

    return np.asarray(list(float_iter), dtype=np.float64)


