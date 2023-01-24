"""
Minimal set of 4x4 column-major matrix functions for building transforms 
compatible with the animation transform API. This serves as reference 
implementation for the different languages we will support so only basic
types and no external libraries are used. 

    [sx, 10, 20, tx]   [x]
    [01, sy, 21, ty] . [y]
    [02, 12, sz, tz]   [z]
    [03, 13, 23, 33]   [1]

"""
import math
from typing import List

Matrix = List[List[float]]

identity = [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]

def multiply(a: Matrix, b: Matrix) -> Matrix:
    assert len(a) == len(b) == 4
    c = [[0., 0., 0., 0.],
         [0., 0., 0., 0.],
         [0., 0., 0., 0.],
         [0., 0., 0., 0.]]
    for row in range(4):
        for col in range(4):
            for k in range(4):
                c[row][col] += a[row][k] * b[k][col]
    return c

def projection_fov(fov_y: float, aspect: float, near: float, far: float) -> Matrix:
    min_x, min_y = -1, -1
    max_x, max_y = 1, 1
    h1 = (max_y + min_y) / (max_y - min_y)
    w1 = (max_x + min_x) / (max_x - min_x)
    t = math.tan(fov_y / 2)
    s1 = 1 / t
    s2 = 1 / (t * aspect)

    # map z to the range [0, 1]
    f1 =  far / (far - near)
    f2 = -(far * near) / (far - near)

    return [[s1,   0.,   w1,  0.],
            [0.,   s2,   h1,  0.],
            [0.,   0.,   f1,  f2],
            [0.,   0.,   1.,  0.]]

def rotation_euler(x: float, y: float, z: float) -> Matrix:
    """Returns a rotation matrix for the given Euler angles (in radians) using XYZ order."""
    a, b = math.cos(x), math.sin(x)
    c, d = math.cos(y), math.sin(y)
    e, f = math.cos(z), math.sin(z)

    ae = a * e 
    af = a * f 
    be = b * e 
    bf = b * f

    return [[ c * e, af + be * d, bf - ae * d, 0.],
            [-c * f, ae - bf * d, be + af * d, 0.],
            [     d,      -b * c,       a * c, 0.],
            [    0.,          0.,          0., 1.]]

def scale(sx: float, sy: float, sz: float) -> Matrix:
    return [[sx, 0., 0., 0.],
            [0., sy, 0., 0.],
            [0., 0., sz, 0.],
            [0., 0., 0., 1.]]

def translation(tx: float, ty: float, tz: float) -> Matrix:
    return [[1., 0., 0., tx],
            [0., 1., 0., ty],
            [0., 0., 1., tz],
            [0., 0., 0., 1.]]
