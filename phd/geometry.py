from typing import Optional
import numpy as np
from .globals import EPS
from .statistics import norm
from .statistics import Sampler


def rmtx_x(angle: float) -> np.ndarray:
    c = np.cos(angle)
    s = np.sin(angle)
    mtx = np.eye(3, dtype=np.float32)
    mtx[1, 1] = c
    mtx[1, 2] = s
    mtx[2, 1] = -s
    mtx[2, 2] = c
    return mtx


def rmtx_y(angle: float) -> np.ndarray:
    c = np.cos(angle)
    s = np.sin(angle)
    mtx = np.eye(3, dtype=np.float32)
    mtx[0, 0] = c
    mtx[0, 2] = -s
    mtx[2, 0] = s
    mtx[2, 2] = c
    return mtx


def rmtx_z(angle: float) -> np.ndarray:
    c = np.cos(angle)
    s = np.sin(angle)
    mtx = np.eye(3, dtype=np.float32)
    mtx[0, 0] = c
    mtx[0, 1] = s
    mtx[1, 0] = -s
    mtx[1, 1] = c
    return mtx


def lookat(location: np.ndarray, target: np.ndarray, dtype=np.float32) -> np.ndarray:
    """
    Compute the pose matrix (rotation, translation)
    of the camera which position is "location" and
    looks at "target" coordinates.
    X axis is parallel to the XY (world) plane
    Z axis points to "target".
    Y is the cross product of Z and X so that it point
    to the positive z-hemispace

    Parameters
    ----------
    location : numpy.ndarray
        Array with 3 values
    target : numpy.ndarray
        Array with 3 values
    dtype : type (default: numpy.float32)
        Desired output data type (e.g. np.float32)

    Returns
    -------
    posemtx : numpy.ndarray
        4x4 pose matrix
    """

    world_z = np.asarray([0, 0, 1])

    cam_z = target - location
    cam_z = cam_z / (norm(cam_z) + EPS)

    cam_x = np.cross(cam_z, world_z)
    cam_x = cam_x / (norm(cam_x) + EPS)

    cam_y = np.cross(cam_z, cam_x)
    cam_y = cam_y / (norm(cam_y) + EPS)

    R = np.stack((cam_x, cam_y, cam_z)).T
    t = location

    posemtx = np.zeros((4, 4))
    posemtx[0:3, 0:3] = R
    posemtx[0:3, 3] = t
    posemtx[3, 3] = 1

    return posemtx.astype(dtype)


def depth_to_pcd(image: np.ndarray, camera_mtx: np.ndarray, camera_pose: np.ndarray = np.eye(4), n: Optional[int] = None, return_coords: bool = False) -> np.ndarray:
    coords = np.stack(np.where(image > 0), axis=-1)
    coords = Sampler.sample(coords, n, "random")
    pu, pv = coords[:, 0], coords[:, 1]
    ones = np.ones((coords.shape[0], 1))
    pcd = (np.linalg.inv(camera_mtx) @ np.stack((pv, pu, ones.reshape(-1)))).T
    pcd *= image[pu, pv].reshape(-1, 1)
    pcd = (np.linalg.inv(camera_pose) @ np.concatenate((pcd, ones), -1).T).T
    pcd = pcd[:, :3].astype(np.float32)
    if return_coords:
        return pcd, (pu, pv)
    return pcd
