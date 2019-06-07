from functools import partial
from itertools import islice
import json
import logging
from operator import attrgetter, methodcaller
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, Optional, Sequence, TypeVar, Union

import spacy
from spacy.language import Language
from spacy.tokens import Doc
import textacy
from textacy import Corpus
from textacy.corpus import _get_spacy_lang
import toolz as tlz
import toolz.curried as tlzc

T = TypeVar('T')

LOGGER = logging.getLogger(__name__)

class TextCorpora:
    '''
    Container for a collection of named textacy corpora
    '''
    def __init__(self,
                 lang: Union[str, Language],
                 corpora: Optional[Dict[str, Corpus]] = None,
                 meta: Optional[dict] = None
                 ):
        self.lang = _get_spacy_lang(lang)
        self._corpora: Dict[str, Corpus] = corpora or {}
        self.meta = meta or {}

    def __repr__(self):
        return "TextCorpora({} corpora, {} docs, {} tokens)"\
            .format(self.n_corpora, self.n_docs(), self.n_tokens())

    def __len__(self):
        return self.n_docs()

    def __iter__(self) -> Doc:
        for corpus in self._corpora.values():
            yield from corpus

    def __contains__(self, doc):
        return any((doc in corpus for corpus in self._corpora.values()))

    def __getitem__(self, idx_or_slice):
        if isinstance(idx_or_slice, str): # one category only
            result = self._corpora[idx_or_slice]
        elif isinstance(idx_or_slice, tuple): # category and indices
            cat, corpus_idx_or_slice = idx_or_slice
            result = self._corpora[cat][corpus_idx_or_slice]
        elif isinstance(idx_or_slice, int): # single item
            if idx_or_slice < 0:
                seq = reversed(self)
                idx_or_slice *= -1
            else:
                seq = iter(self)
            result = tlz.nth(idx_or_slice, seq)
        elif isinstance(idx_or_slice, slice):
            result = list(islice(iter(self),
                                 idx_or_slice.start,
                                 idx_or_slice.stop,
                                 idx_or_slice.step))
        else:
            raise KeyError(f'invalid index {idx_or_slice}')

        return result

    @property
    def n_corpora(self):
        return len(self._corpora)

    @property
    def labels(self):
        return [c for c in self._corpora.keys()]

    def n_docs(self, *, corpora: Optional[Union[str, Sequence[str]]] = None):
        func = attrgetter('n_docs')

        return self._agg_with(func, sum, corpora)

    def n_sents(self, *, corpora: Optional[Union[str, Sequence[str]]] = None):
        func = attrgetter('n_sents')

        return self._agg_with(func, sum, corpora)

    def n_tokens(self, *, corpora: Optional[Union[str, Sequence[str]]] = None):
        func = attrgetter('n_tokens')

        return self._agg_with(func, sum, corpora)

    def word_counts(
            self,
            normalize: str = 'lemma',
            weighting: str = 'count',
            as_strings: bool = False,
            *,
            corpora: Optional[Union[str, Sequence[str]]] = None
    ) -> dict:
        '''
        Map the set of unique words in :class:`Corpus` to their counts as
        absolute, relative, or binary frequencies of occurence,
        similar to :meth:`Doc._.to_bag_of_words()` but aggregated over all corpora an docs.
        '''
        func = methodcaller('word_counts', normalize, weighting, as_strings)

        return self._agg_with(func, tlz.merge_with(sum), corpora)

    def word_doc_counts(
            self,
            normalize: str = 'lemma',
            weighting: str = 'count',
            smooth_idf: bool = True,
            as_strings: bool = False,
            *,
            corpora: Optional[Union[str, Sequence[str]]] = None
    ) -> dict: # pylint: disable=too-many-arguments
        '''
        Map the set of unique words in the Corpora to their document counts as absolute,
        relative, inverse, or binary frequencies of occurence.
        '''
        func = methodcaller('word_doc_count', normalize, weighting, smooth_idf, as_strings)

        return self._agg_with(func, tlz.merge_with(sum), corpora)

    def _agg_with(
            self,
            func: Callable[[Corpus], T],
            merge_func: Callable[[Iterable[T]], T],
            corpora: Optional[Union[str, Sequence[str]]] = None
    ) -> T:
        '''
        Delegates the calculation of basic corpus statistics to the corpora.
        '''

        aggs: Iterable[T] = (func(corpus) for corpus in  self.corpora(corpora))
        agg = merge_func(aggs)

        return agg

    def corpora(self, corpora: Optional[Union[str, Sequence[str]]] = None) -> Iterator[Corpus]:
        '''
        Selects one or more corpus from the corpora. Gets all the corpora by default.
        '''
        matched_corpora: Iterable[Corpus]

        if isinstance(corpora, str):
            matched_corpora = [self._corpora[corpora]]
        elif isinstance(corpora, Sequence):
            matched_corpora = (
                corpus

                for cat, corpus in self._corpora.values()

                if cat in corpora
            )
        else:
            matched_corpora = (
                corpus

                for corpus in self._corpora.values()
            )

        yield from matched_corpora

    def get(self,
            match_func: Callable[[Doc], bool],
            limit: Optional[int] = None,
            corpora: Optional[Union[str, Sequence[str]]] = None
            ) -> Iterator[Doc]:
        '''
         Get all (or N <= ``limit``) docs in :class:`TextCorpora` for which
        ``match_func(doc)`` is True.
        '''

        matched_corpora = self.corpora(corpora)
        matched_docs = tlz.concat(corpus.get(match_func) for corpus in matched_corpora)

        for doc in islice(matched_docs, limit):
            yield doc

    def remove(self,
               match_func: Callable[[Doc], bool],
               limit: Optional[int] = None,
               corpora: Optional[Union[str, Sequence[str]]] = None
               ) -> None:
        '''
        Remove all (or N <= ``limit`` per :class:`Corpus`) docs in :class:`TextCorpora` for which
        ``match_func(doc)`` is True. Corpus doc/sent/token counts are adjusted
        accordingly.
        '''

        matched_corpora = self.corpora(corpora)

        for corpus in matched_corpora:
            corpus.remove(match_func, limit)

    def save(self, dirpath: Union[str, Path]):
        '''
        Saves the corpora binaries and the :class:`TextCorpora` metadata to the directory specified by dirpath.
        '''
        dirpath = Path(dirpath)

        if not (dirpath.exists() and dirpath.is_dir()):
            raise FileNotFoundError(f'dirpath {dirpath} must be an existing directory')

        with dirpath.joinpath('metadata.json').open('wt') as f:
            json.dump(self.meta, f)

        for label, corpus in self._corpora.items():
            filepath = dirpath.joinpath(f'{label}.bin').as_posix()
            corpus.save(filepath)

    @classmethod
    def load(cls, lang: Union[str, Language], dirpath: Union[str, Path]):
        '''
        Loads the corpora binaries and the :class:`TextCorpora` metadata from the directory specified by dirpath.
        '''
        dirpath = Path(dirpath)

        if not (dirpath.exists() and dirpath.is_dir()):
            raise FileNotFoundError(f'dirpath {dirpath} must be an existing directory')

        spacy_lang = _get_spacy_lang(lang)

        with dirpath.joinpath('metadata.json').open('rt') as f:
            meta = json.load(f)

        corpora = {
            filepath.stem: Corpus.load(spacy_lang, filepath.as_posix())

            for filepath in dirpath.glob('*.bin')
        }

        return cls(spacy_lang, corpora, meta)

    def add_corpus(self, label: str, corpus: Corpus):
        if label in self._corpora:
            LOGGER.warning('corpus %s exists. It will be overwritten', label)
        self._corpora[label] = corpus


    # def add():
    #     pass

    # def add_doc():
    #     pass

    # def add_docs():
    #     pass

    # def add_record():
    #     pass

    # def add_records():
    #     pass

    # def add_text():
    #     pass

    # def add_texts():
    #     pass

    # def vector_norms():
    #     pass

    # def vectors():
    #     pass



__all__ = [
    'TextCorpora'
]