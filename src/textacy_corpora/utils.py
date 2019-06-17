import os
from pathlib import Path
from typing import Union

from .typing import Datapoint, LabelledDatapoint


def splitext(basename: Union[str, Path]):
    stem, ext = os.path.splitext(basename)

    if ext == '':
        result = stem
    else:
        result = splitext(stem)

    return result


def unpack(labelled_data: LabelledDatapoint) -> Datapoint:
    data: Datapoint

    if len(labelled_data) > 2:
        data = labelled_data[1:]
    else:
        data = labelled_data[-1]

    return data
