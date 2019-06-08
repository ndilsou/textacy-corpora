from typing import Callable, Dict, Iterable, Iterator, Tuple, Union, Sequence

from spacy.tokens import Doc
import toolz as tlz

from textacy_corpora.typing import LabelledData


def to_labelled(collection: Dict[str, Sequence]) -> Iterator[LabelledData]:
    seq = flatten_collection(collection)
    (_, first), seq = tlz.peek(seq)

    labeller: Callable

    if isinstance(first, tuple):
        labeller = _from_record
    else:
        labeller = _from_scalar

    for label, data in seq:
        yield labeller(label, data)


def flatten_collection(collection: Dict[str, Sequence]) -> Iterator:
    for label, seq in collection.items():
        for item in seq:
            yield (label, item)


def _from_scalar(label: str, data: Union[str, Doc]) -> Tuple[str, Union[str, Doc]]:
    return label, data


def _from_record(label: str, data: Tuple[str, dict]) -> Tuple[str, str, dict]:
    return (label, *data)
