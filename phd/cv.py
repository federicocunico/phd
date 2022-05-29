import logging
from typing import Any, Generator, Optional
from enum import Enum, auto
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
