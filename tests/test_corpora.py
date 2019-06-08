from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pytest
import spacy
from spacy.language import Language
from spacy.tokens import Doc
from textacy import Corpus
import toolz as tlz

import textacy_corpora
from textacy_corpora import TextCorpora

from .utils import to_labelled


def test_can_create_empty_corpora(spacy_lang: Language) -> None:
    corpora = TextCorpora(spacy_lang)
    assert isinstance(corpora, TextCorpora)


def test_can_create_corpora(spacy_lang: Language, corpus_collection: Dict[str, Corpus]) -> None:
    corpora = TextCorpora(spacy_lang, corpus_collection)
    assert isinstance(corpora, TextCorpora)
    assert corpora.labels == ['cat1', 'cat2']
    assert corpora.n_corpora == 2


def test_corpora_length(text_corpora: TextCorpora) -> None:
    assert len(text_corpora) == 4


def test_n_docs_attributes(text_corpora: TextCorpora) -> None:
    assert text_corpora.n_docs() == 4
    assert text_corpora.n_docs('cat1') == 2
    assert text_corpora.n_docs('cat2') == 2
    assert text_corpora.n_docs(['cat2', 'cat1']) == 4


def test_n_sents_attributes(text_corpora: TextCorpora) -> None:
    assert text_corpora.n_sents() == 6
    assert text_corpora.n_sents('cat1') == 2
    assert text_corpora.n_sents('cat2') == 4
    assert text_corpora.n_sents(['cat2', 'cat1']) == 6


def test_n_tokens_attributes(text_corpora: TextCorpora) -> None:
    assert text_corpora.n_tokens() == 51
    assert text_corpora.n_tokens('cat1') == 28
    assert text_corpora.n_tokens('cat2') == 23
    assert text_corpora.n_tokens(['cat2', 'cat1']) == 51


def test_word_counts_attributes(text_corpora: TextCorpora, corpus_collection: Dict[str, Corpus]) -> None:
    wc_cat1 = corpus_collection['cat1']\
                             .word_counts(as_strings=True)
    wc_cat2 = corpus_collection['cat2']\
                             .word_counts(as_strings=True)
    wc = tlz.merge_with(sum, wc_cat1, wc_cat2)
    assert text_corpora.word_counts(corpora='cat1', as_strings=True) == wc_cat1
    assert text_corpora.word_counts(corpora='cat2', as_strings=True) == wc_cat2
    assert text_corpora.word_counts(as_strings=True) == wc
    assert text_corpora.word_counts(corpora=['cat2', 'cat1'], as_strings=True) == wc


def test_word_doc_counts_attributes(text_corpora: TextCorpora, corpus_collection: Dict[str, Corpus]) -> None:
    wdc_cat1 = corpus_collection['cat1']\
                             .word_doc_counts(as_strings=True)
    wdc_cat2 = corpus_collection['cat2']\
                             .word_doc_counts(as_strings=True)
    wdc = tlz.merge_with(sum, wdc_cat1, wdc_cat2)
    assert text_corpora.word_doc_counts(corpora='cat1', as_strings=True) == wdc_cat1
    assert text_corpora.word_doc_counts(corpora='cat2', as_strings=True) == wdc_cat2
    assert text_corpora.word_doc_counts(as_strings=True) == wdc
    assert text_corpora.word_doc_counts(corpora=['cat2', 'cat1'], as_strings=True) == wdc


def test_indexing(text_corpora: TextCorpora, corpus_collection: Dict[str, Corpus]) -> None:
    assert text_corpora[0] == corpus_collection['cat1'][0]
    assert text_corpora[-1] == corpus_collection['cat2'][-1]
    assert text_corpora[1:4] == [corpus_collection['cat1'][1], *corpus_collection['cat2'][:2]]
    assert text_corpora['cat1', 1] == corpus_collection['cat1'][1]
    assert text_corpora['cat2', :] == corpus_collection['cat2'][:]


def test_can_add_corpus(spacy_lang: Language, corpus_collection: Dict[str, Corpus]) -> None:
    corpora = TextCorpora(spacy_lang)
    corpora.add_corpus('cat1', corpus_collection['cat1'])
    assert corpora['cat1'] == corpus_collection['cat1']


def test_can_add_text(spacy_lang: Language, text_collection: Dict[str, Sequence[str]]) -> None: # pylint: disable=missing-docstring
    corpora = TextCorpora(spacy_lang)

    for text in text_collection['cat1']:
        corpora.add_text('cat1', text)

    for text in text_collection['cat2']:
        corpora.add_text('cat2', text)

    assert len(corpora) == 4
    assert corpora.labels == ['cat1', 'cat2']

def test_can_add_texts(spacy_lang: Language, text_collection: Dict[str, Sequence[str]]) -> None: # pylint: disable=missing-docstring
    corpora = TextCorpora(spacy_lang)

    corpora.add_texts('cat1', text_collection['cat1'])
    corpora.add_texts('cat2', text_collection['cat2'])

    assert len(corpora) == 4
    assert corpora.labels == ['cat1', 'cat2']

def test_can_add_doc(spacy_lang: Language, doc_collection: Dict[str, Sequence[Doc]]) -> None: # pylint: disable=missing-docstring
    corpora = TextCorpora(spacy_lang)

    for doc in doc_collection['cat1']:
        corpora.add_doc('cat1', doc)

    for doc in doc_collection['cat2']:
        corpora.add_doc('cat2', doc)

    assert len(corpora) == 4
    assert corpora.labels == ['cat1', 'cat2']


def test_can_add_docs(spacy_lang: Language, doc_collection: Dict[str, Sequence[Doc]]) -> None: # pylint: disable=missing-docstring
    corpora = TextCorpora(spacy_lang)

    corpora.add_docs('cat1', doc_collection['cat1'])
    corpora.add_docs('cat2', doc_collection['cat2'])

    assert len(corpora) == 4
    assert corpora.labels == ['cat1', 'cat2']


def test_can_add_record(spacy_lang: Language, record_collection: Dict[str, Sequence[str]]) -> None: # pylint: disable=missing-docstring
    corpora = TextCorpora(spacy_lang)

    for record in record_collection['cat1']:
        corpora.add_record('cat1', record)

    for record in record_collection['cat2']:
        corpora.add_record('cat2', record)

    assert len(corpora) == 4
    assert corpora.labels == ['cat1', 'cat2']


def test_can_add_records(spacy_lang: Language, record_collection: Dict[str, Sequence[str]]) -> None: # pylint: disable=missing-docstring
    corpora = TextCorpora(spacy_lang)

    corpora.add_records('cat1', record_collection['cat1'])
    corpora.add_records('cat2', record_collection['cat2'])

    assert len(corpora) == 4
    assert corpora.labels == ['cat1', 'cat2']


def test_add_with_texts(spacy_lang: Language, text_collection: Dict[str, Sequence[str]]) -> None: # pylint: disable=missing-docstring
    corpora = TextCorpora(spacy_lang)

    corpora.add('cat1', text_collection['cat1'])
    corpora.add('cat2', text_collection['cat2'])

    assert len(corpora) == 4
    assert corpora.labels == ['cat1', 'cat2']


def test_add_with_docs(spacy_lang: Language, doc_collection: Dict[str, Sequence[str]]) -> None: # pylint: disable=missing-docstring
    corpora = TextCorpora(spacy_lang)

    corpora.add('cat1', doc_collection['cat1'])
    corpora.add('cat2', doc_collection['cat2'])

    assert len(corpora) == 4
    assert corpora.labels == ['cat1', 'cat2']


def test_add_with_records(spacy_lang: Language, record_collection: Dict[str, Sequence[str]]) -> None: # pylint: disable=missing-docstring
    corpora = TextCorpora(spacy_lang)

    corpora.add('cat1', record_collection['cat1'])
    corpora.add('cat2', record_collection['cat2'])

    assert len(corpora) == 4
    assert corpora.labels == ['cat1', 'cat2']


def test_add_labelled_texts(spacy_lang: Language, text_collection: Dict[str, Sequence[str]]) -> None: # pylint: disable=missing-docstring
    corpora = TextCorpora(spacy_lang)

    labelled_collection = to_labelled(text_collection)

    corpora.add_labelled(labelled_collection)

    assert len(corpora) == 4
    assert corpora.labels == ['cat1', 'cat2']


def test_add_labelled_docs(spacy_lang: Language, doc_collection: Dict[str, Sequence[Doc]]) -> None: # pylint: disable=missing-docstring
    corpora = TextCorpora(spacy_lang)

    labelled_collection = to_labelled(doc_collection)

    corpora.add_labelled(labelled_collection)

    assert len(corpora) == 4
    assert corpora.labels == ['cat1', 'cat2']


def test_add_labelled_with_records(spacy_lang: Language, record_collection: Dict[str, Sequence[Tuple[str, dict]]]) -> None: # pylint: disable=missing-docstring
    corpora = TextCorpora(spacy_lang)

    labelled_collection = to_labelled(record_collection)

    corpora.add_labelled(labelled_collection)

    assert len(corpora) == 4
    assert corpora.labels == ['cat1', 'cat2']

def test_vector(text_corpora: TextCorpora, corpus_collection: Dict[str, Corpus]) -> None:
    vector_cat1 = corpus_collection['cat1'].vectors
    vector_cat2 = corpus_collection['cat2'].vectors
    vector = np.vstack((vector_cat1, vector_cat2))

    assert np.all(np.isclose(text_corpora.vectors('cat1'), vector_cat1))
    assert np.all(np.isclose(text_corpora.vectors('cat2'), vector_cat2))
    assert np.all(np.isclose(text_corpora.vectors(), vector))
    assert np.all(np.isclose(text_corpora.vectors(['cat1', 'cat2']), vector))
