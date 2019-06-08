from typing import Union, Tuple, Iterable
from spacy.tokens import Doc

Datapoint = Union[str, Doc, Tuple[str, dict]]
Data = Union[Datapoint, Iterable[Datapoint]]
LabelledDatapoint = Union[Tuple[str, str], Tuple[str, Doc], Tuple[str, str, dict]]
LabelledData = Union[LabelledDatapoint, Iterable[LabelledDatapoint]]
