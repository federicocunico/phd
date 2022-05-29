from typing import List, Optional, Tuple, Union
import numpy as np


def stats(arr: Union[List, np.ndarray], axis: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    m = np.min(arr, axis=axis)
    M = np.max(arr, axis=axis)
    mu = np.mean(arr, axis=axis)
    sd = np.std(arr, axis=axis)
    return m, M, mu, sd


def flip(arr: np.ndarray, axis: int) -> np.ndarray:
    """
    Flip a numpy array.

    Parameters
    ----------
    arr : numpy.ndarray
        Input array
    axis : integer
        Axis along which the array is flipped

    Returns
    -------
    arr : numpy.ndarray
        Flipped version of the input array
    """

    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)

    if isinstance(axis, str):
        axis = axis.lower()
        if axis == "horizontal":
            axis = 0
        if axis == "vertical":
            axis = 1

    arr = np.flip(arr, axis)
    return arr


def addnoise(arr: np.ndarray, mu: float, sd: float) -> np.ndarray:
    """
    Add Gaussian noise to numpy array.

    Parameters
    ----------
    arr : numpy.ndarray
        Input array
    mu : float
        Mean of the noise distribution
    sd : float
        Standard deviation of the noise distribution

    Returns
    -------
    arr : numpy.ndarray
        Perturbed version of the input array (as float32)
    """
    noise = np.random.normal(loc=mu, scale=sd, size=arr.shape)
    return arr.astype(np.float32) + noise.astype(np.float32)


def normalize(arr: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """
    Normalize a vector between 0 and 1.

    Parameters
    ----------
    arr : numpy.ndarray
        Input array
    axis : integer
        Axis along which normalization is computed

    Returns
    -------
    arr : numpy.ndarray
        Normalized version of the input array
    """

    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)

    arr = arr - np.min(arr, axis)
    M = np.max(arr, axis)
    if np.asarray(M).size == 1 and M == 0:
        M = 1
    if np.asarray(M).size > 1:
        M[np.where(M == 0)] = 1
    arr = arr / M

    return arr


def norm(arr: np.ndarray, axis: Optional[int] = None, lp: int = 2) -> np.ndarray:
    """
    Compute the norm of a vector.

    Parameters
    ----------
    arr : tuple, list or numpy.ndarray
        Input array
    axis : integer (default: None, ie all array dimensions)
        Axis along which norm is computed
    lp : positive integer (default: 2, ie Euclidean norm)
        Factor of lp-norm

    Returns
    -------
        lp-norm of the input array
    """
    if lp == 0:
        return np.nonzero(np.asarray(arr))[0].size
    return np.sum(np.asarray(arr)**lp, axis=axis)**(1/lp)


def concatenate(arr1: np.ndarray, arr2: np.ndarray, axis: int = 0, expand: bool = False) -> np.ndarray:
    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)
    if len(arr1) == 0:
        if expand:
            return np.expand_dims(arr2, axis)
        return arr2
    if len(arr2) == 0:
        if expand:
            return np.expand_dims(arr1, axis)
        return arr1
    if arr1.ndim != arr2.ndim:
        if arr1.ndim < arr2.ndim:
            arr1 = np.expand_dims(arr1, axis)
        else:  # arr1.ndim > arr2.ndim
            arr2 = np.expand_dims(arr2, axis)
    return np.concatenate((arr1, arr2), axis)


class Sampler():

    def __init__(self) -> None:
        pass

    @staticmethod
    def downsample(X: np.ndarray, factor: int, mode: str = "uniform") -> np.ndarray:
        """
        Downsample data matrix rows.

        Parameters
        ----------
        X : numpy.ndarray
            Array to downsample
        factor : integer
            Sampling factor

        Returns
        -------
        pts : numpy.ndarray
            Extracted points
        """

        if mode == "uniform":
            ids = np.arange(0, X.shape[0], factor)
        elif mode == "random":
            ids = np.random.permutation(X.shape[0])
            ids = ids[:(X.shape[0] // factor)]
        else:
            raise NotImplementedError()
        pts = X[ids]

        return pts

    @staticmethod
    def sample(X: np.ndarray, n: int, mode: str = "uniform") -> np.ndarray:
        """
        Downsample data matrix rows.

        Parameters
        ----------
        X : numpy.ndarray
            Array to downsample
        factor : integer
            Sampling factor

        Returns
        -------
        pts : numpy.ndarray
            Extracted points
        """

        if n == None:
            return X

        if mode == "uniform":
            factor = X.shape[0] // n
            return Sampler.downsample(X, factor)
        elif mode == "random":
            ids = np.random.permutation(X.shape[0])
            ids = ids[:n]
        else:
            raise NotImplementedError()
        pts = X[ids]

        return pts

    @staticmethod
    def sample_primitive(surf: str = "sphere", n: int = 100) -> np.ndarray:
        """
        Sample the surface of a primitive shape.

        Parameters
        ----------
        surf : string (default: "sphere")
            Type of the surface to sample. Currently supported:
            - sphere
        n : integer (default: 100)
            Number of samples

        Returns
        -------
        pts : numpy.ndarray
            Array of positions with shape nx3
        """

        pts = None

        if surf == "sphere":
            pts = Sampler._sample_sphere(n)
        else:
            raise NotImplementedError()

        return pts

    @staticmethod
    def _sample_sphere(n: int) -> np.ndarray:
        """
        Distributing many points on a sphere.
        SAFF, Edward B.; KUIJLAARS, Amo BJ.
        The mathematical intelligencer, 1997, 19.1: 5-11.

        Parameters
        ----------
        n : integer
            Number of samples

        Returns
        -------
        pts : numpy.ndarray
            Array of positions with shape nx3
        """

        if n < 1:
            return None

        s = 3.6 / np.sqrt(n)
        dz = 2.0 / n
        angle = 0.0
        z = 1 - dz / 2

        pts = np.zeros((n, 3), dtype=np.float32)

        for i in range(n):
            r = np.sqrt(1 - z * z)
            # compute coordinates
            x = np.cos(angle) * r
            y = np.sin(angle) * r
            pts[i] = [x, y, z]
            # update
            angle = angle + s / r
            z = z - dz

        return pts
