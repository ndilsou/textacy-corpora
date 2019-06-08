from typing import Dict, List, Tuple, Sequence

import pytest
import spacy
from spacy.tokens import Doc
from spacy.language import Language
from textacy import Corpus

from textacy_corpora import TextCorpora


@pytest.fixture(scope='session')
def spacy_lang() -> Language:
    return spacy.load('en_core_web_sm')


@pytest.fixture
def text_collection() -> Dict[str, List[str]]:
    return {
        'cat1': [
            'I found a parking place half a block away, sat in the car and waited.',
            'My quarry was in the apartment house for two hours.'
        ],
        'cat2': [
            'This document is a test. You should know this.',
            'The beginning of a great story. Nobody believed me though.'
        ]
    }



@pytest.fixture
def record_collection(text_collection: Dict[str, Sequence[str]]) -> Dict[str, List[Tuple[str, dict]]]:
    return {
        'cat1': [
            (text_collection['cat1'][0], {'corpus': 'cat1', 'title': 'text1'}),
            (text_collection['cat1'][1], {'corpus': 'cat1', 'title': 'text2'})
        ],
        'cat2': [
            (text_collection['cat2'][0], {'corpus': 'cat1', 'title': 'text3'}),
            (text_collection['cat2'][1], {'corpus': 'cat1', 'title': 'text4'})
        ]
    }


@pytest.fixture
def doc_collection(spacy_lang: Language, text_collection: Dict[str, Sequence[str]]) -> Dict[str, List[Doc]]:
    return {
        'cat1': [spacy_lang(t) for t in text_collection['cat1']],
        'cat2': [spacy_lang(t) for t in text_collection['cat2']],
    }


@pytest.fixture
def corpus_collection(spacy_lang: Language, text_collection: Dict[str, Sequence[str]]) -> Dict[str, Corpus]:
    return {
        'cat1': Corpus(spacy_lang, text_collection['cat1']),
        'cat2': Corpus(spacy_lang, text_collection['cat2'])
    }


@pytest.fixture
def text_corpora(spacy_lang: Language, corpus_collection: Dict[str, Corpus]) -> TextCorpora:
    return TextCorpora(spacy_lang, corpus_collection)
