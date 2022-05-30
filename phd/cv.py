import logging
from typing import Any, Generator, List, Optional, Tuple, Union
from enum import Enum, auto
from uuid import UUID
import cv2
from PIL import Image

from .io_utils import get_files_from


class ImageFormat(Enum):
    PIL = auto()
    NDARRAY = auto()
    STRING = auto()


def folder_reader(path: str, image_extension: Optional[str] = None, format: ImageFormat = ImageFormat.PIL) -> Generator[int, str, Any]:
    files = get_files_from(path, image_extension)

    for i, file in enumerate(files):
        image = None
        if format == ImageFormat.PIL:
            try:
                image = Image.open(file)
                image = image.convert("RGB")
            except:
                logging.error(
                    f"Unable to read the image: {file}. Returning None")
        elif format == ImageFormat.NDARRAY:
            image = cv2.imread(file)
        elif format == ImageFormat.STRING:
            image = file
        else:
            raise NotImplementedError(
                f"Invalid ImageFormat option provided: {format.name}")

        res = (i, file, image)
        yield res


def unique_color(val: Union[str, UUID, Tuple, List, int]) -> Tuple[int, int, int]:
    if isinstance(val, str):
        val = UUID(val)
    if isinstance(val, (int, float)):
        val = tuple([val]*10)
    if isinstance(val, (UUID, Tuple, List)):
        hash_val = val.__hash__()
    else:
        raise NotImplementedError("No valid type provided")

    color_code = abs(hash_val) % 0x1000000
    red = color_code >> 16
    green = (color_code >> 8) & 0xFF
    blue = color_code & 0xFF
    # return float(red) // 256, float(green) // 256, float(blue) // 256
    return int(red), int(green), int(blue)
