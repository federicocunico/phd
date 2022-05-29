from typing import List, Tuple
import numpy as np
from .geometry import rmtx_x, rmtx_y, rmtx_z


def rotation_to_euler(R: np.ndarray, mode: str = "XYZ") -> Tuple[float, float, float]:
    if mode == "XYZ":
        theta1 = np.math.atan2(R[1, 2], R[2, 2])
        c2 = np.sqrt(R[0, 0]**2 + R[0, 1]**2)
        theta2 = np.math.atan2(-R[0, 2], c2)
        s1 = np.math.sin(theta1)
        c1 = np.math.cos(theta1)
        theta3 = np.math.atan2(
            s1*R[2, 0] - c1*R[1, 0], c1*R[1, 1] - s1*R[2, 1])
    else:
        raise NotImplementedError()
    return [theta1, theta2, theta3]


def euler_to_rotation(theta1: float, theta2: float, theta3: float, mode: str = "XYZ") -> np.ndarray:
    if mode == "XYZ":
        Rx = rmtx_x(theta1)
        Ry = rmtx_y(theta2)
        Rz = rmtx_z(theta3)
        R = np.matmul(Rx, np.matmul(Ry, Rz))
    else:
        raise NotImplementedError()
    return R


def quaternion_to_rotation(q: List[float]) -> np.ndarray:
    assert len(q) == 4, "Quaternion should have 4 elements"
    w, x, y, z = q

    r11 = w*w + x*x - y*y - z*z
    r12 = 2*(x*y - w*z)
    r13 = 2*(w*y + x*z)

    r21 = 2*(x*y + w*z)
    r22 = w*w - x*x + y*y - z*z
    r23 = 2*(y*z - w*x)

    r31 = 2*(x*z - w*y)
    r32 = 2*(y*z + w*x)
    r33 = w*w - x*x - y*y + z*z

    return np.array([
        [r11, r12, r13],
        [r21, r22, r23],
        [r31, r32, r33],
    ])
