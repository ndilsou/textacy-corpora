from .typing import Datapoint, LabelledDatapoint


def unpack(labelled_data: LabelledDatapoint) -> Datapoint:
    data: Datapoint

    if len(labelled_data) > 2:
        data = labelled_data[1:]
    else:
        data = labelled_data[-1]

    return data
