from dataclasses import dataclass
from typing import NamedTuple, Union

import spacy.tokens
from spacy.matcher import DependencyMatcher
from spacy.symbols import *
from spacy.tokens import *

from posextract.verb_phrase import VerbPhrase, ADVCLVerbPhrase, ConjVerbPhrase, CCompVerbPhrase, \
    add_verb_phrase_patterns

__DEP_MATCHER = None
__NLP = None


def get_nlp():
    global __NLP
    if __NLP is None:
        __NLP = spacy.load("en_core_web_sm")
    return __NLP


def get_dep_matcher(nlp):
    global __DEP_MATCHER

    if __DEP_MATCHER is None:
        matcher = DependencyMatcher(get_nlp().vocab)
        add_verb_phrase_patterns(matcher)
        __DEP_MATCHER = matcher

    return __DEP_MATCHER


def should_consider_verb_phrase(verb_phrase: VerbPhrase):
    if isinstance(verb_phrase, CCompVerbPhrase):
        return True

    for child in verb_phrase.second.children:
        if child.dep == nsubj or child.dep == nsubjpass:
            return False

    return True


class TripleExtractorOptions(NamedTuple):
    compound_subject: bool = True
    compound_object: bool = True
    combine_adj: bool = False
    add_auxiliary: bool = False
    prep_phrase: bool = False
    lemmatize: bool = False


VERB_DEP_TAGS = {ccomp, relcl, xcomp, acl, advcl, pcomp, csubj, csubjpass, conj}
OBJ_DEP_TAGS = {dobj, pobj, acomp}  # dative?


def is_root(token: Token):
    return token.dep_ == 'ROOT'


def is_verb(token: Token):
    if token.dep_ == 'ROOT':
        return True

    if token.pos == PROPN and token.dep == conj:
        return False

    return token.dep in VERB_DEP_TAGS


def is_object(token: Token):
    if token.pos == NOUN and token.dep == amod:
        return True

    if token.pos == NOUN and token.dep == attr:
        return True

    if token.pos == PROPN and token.dep == attr:
        return True

    if token.pos == ADV and token.dep == advmod:
        return True

    if token.pos == PRON and token.dep_ == "dative":
        return True

    return token.dep in OBJ_DEP_TAGS


def is_noun_attribute(token: Token):
    return (token.pos == NOUN or token.pos == PROPN) and token.dep == attr


def is_poa(token: Token):
    return token.dep == prep or token.dep == agent or token.dep == det or token.dep == nmod


def get_verb_neg(token: Union[Token, VerbPhrase], up=True):
    if isinstance(token, VerbPhrase):
        children = token.second.children
        verb_parent = token.second.head
    else:
        children = token.children
        verb_parent = token.head

    for child in children:
        if child.dep == neg:
            return child, None

    if verb_parent.pos == VERB and verb_parent.text.lower() == 'failed' and token.dep == xcomp:
        try:
            child = next(children)
            if child.pos == PART and child.text.lower() == 'to':
                return verb_parent, child
        except StopIteration:
            return None, None
    elif verb_parent.pos == VERB and (token.dep == ccomp or token.dep == xcomp):
        for child in verb_parent.children:
            if child.dep == neg:
                return child, None

    # if up and token.head.pos == VERB:
    #     parent_negation = get_verb_neg(token.head, up=False)
    #     if parent_negation:
    #         return parent_negation

    return None, None


def get_subject_neg(token):
    for child in token.children:
        if child.dep == det and child.text.lower() in ("no", "not", "never"):
            return child
        if child.dep == neg:
            return child

    return None


def get_poa_neg(token):
    for child in token.children:
        if child.dep == neg:
            return child

    return None


def get_object_neg(token):
    for child in token.children:
        if child.dep == det and child.text.lower() in ("no", "not", "never"):
            return child

        if child.dep == neg:
            return child

    if token.head.pos == PART and token.head.text.lower() in ("not", ):
        return token.head

    return None


def get_object_adj(token):
    for child in token.children:
        if child.dep == amod and child.pos == ADJ:
            return child
        if child.dep == advmod and child.pos == ADV:
            return child

    return None
