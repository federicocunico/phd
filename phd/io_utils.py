import os
import pickle
import re
from glob import glob
from typing import Any, Iterable, List, Optional


def natural_sort(l: Iterable) -> Iterable:
    def convert(text): return int(text) if text.isdigit() else text
    def alphanum_key(key): return [convert(c)
                                   for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def get_files_from(path: str, extension: Optional[str] = None, recursive: bool = False) -> List[str]:

    if extension is None:
        ext = ""
    else:
        ext = extension if extension[0] == "." else f".{extension}"

    if recursive:
        path = os.path.join(path, "**", f"*{ext}")
    else:
        path = os.path.join(path, f"*{ext}")

    files = natural_sort(glob(path))
    return files


def read_pickle(pickle_path: str) -> Any:
    with open(pickle_path, "rb") as fp:
        data = pickle.load(fp)
    return data


def write_pickle(data, pickle_path: str) -> None:
    with open(pickle_path, "wb") as fp:
        pickle.dump(data, fp)
