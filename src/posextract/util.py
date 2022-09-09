from spacy.symbols import *
from spacy.tokens import Token
from typing import NamedTuple, Optional, List
from dataclasses import dataclass
import dataclasses
import spacy.tokens
import copy


VERB_DEP_TAGS = {ccomp, relcl, xcomp, acl, advcl, pcomp, csubj, csubjpass, conj}
OBJ_DEP_TAGS = {dobj, pobj, acomp} # dative?


@dataclass
class TripleExtractionFlattened:
    subject_negdat: str = ''
    subject: str = ''
    neg_adverb: str = ''
    aux_verb: str = ''
    verb: str = ''
    poa: str = ''
    object_negdat: str = ''
    adjectives: str = ''
    object: str = ''

    def astuple(self):
        return (v for v in self.__dict__.values())

    def __str__(self):
        return ' '.join((str(v) for v in self.astuple() if v))


@dataclass
class TripleExtraction:
    subject_negdat: Optional[Token] = None
    subject: Optional[Token] = None
    neg_adverb: Optional[Token] = None
    aux_verb: Optional[Token] = None
    verb: Optional[Token] = None
    poa: Optional[Token] = None
    object_negdat: Optional[Token] = None
    adjectives: Optional[List[Token]] = None
    object: Optional[Token] = None

    def flatten(self, lemmatize=False) -> TripleExtractionFlattened:
        kwargs = {k: v for k, v in self.__dict__.items() if v is not None}

        if lemmatize:
            if self.object:
                kwargs['object'] = self.object.lemma_
            if self.verb:
                kwargs['verb'] = self.verb.lemma_
            if self.subject:
                kwargs['subject'] = self.subject.lemma_

        if self.adjectives:
            kwargs['adjectives'] = ' '.join((adj.text for adj in self.adjectives))

        for k, v in kwargs.items():
            if type(v) != str:
                kwargs[k] = str(v)

        return TripleExtractionFlattened(
            **kwargs
        )


def is_root(token: Token):
    return token.dep_ == 'ROOT'


def is_verb(token: Token):
    if token.dep_ == 'ROOT':
        return True

    return token.dep in VERB_DEP_TAGS


def is_object(token: Token):
    if token.pos == NOUN and token.dep == amod:
        return True

    if token.pos == ADV and token.dep == advmod:
        return True

    if token.pos == PRON and token.dep_ == "dative":
        return True

    return token.dep in OBJ_DEP_TAGS


def is_noun_attribute(token: Token):
    return token.pos == NOUN and token.dep == attr


def is_poa(token: Token):
    return token.dep == prep or token.dep == agent or token.dep == det or token.dep == nmod


def object_search(token: Token):
    objects = []

    visited = set()
    considering = [token, ]

    while considering:
        candidate = considering.pop(-1)

        if candidate in visited:
            continue

        visited.add(candidate)

        if is_object(candidate):
            obj_negdat = get_object_neg(candidate)
            # obj_adj = get_object_adj(candidate)
            poa = candidate.head if is_poa(candidate.head) else None
            objects.append((poa, obj_negdat, candidate))

        for child in candidate.children:
            if child not in visited:
                if child.pos == VERB:
                    continue
                considering.append(child)

    return objects


def subject_search(token: Token):
    objects = []

    visited = set()
    considering = [token, ]

    # print('Doing subject search for token: ', token)
    # print('verb.head', token.head)
    # print('verb.children', list(token.children))

    while considering:
        candidate = considering.pop(-1)

        if candidate in visited:
            continue

        visited.add(candidate)

        if candidate.dep == nsubj or candidate.dep == nsubjpass:
            objects.append((get_subject_neg(candidate), candidate))

        for child in candidate.children:
            if child not in visited:
                if child.pos == VERB:
                    continue
                considering.append(child)

        parent = candidate.head
        if parent not in visited:
            considering.append(parent)

    return objects


def get_verb_neg(token):
    for child in token.children:
        if child.dep == neg:
            return child

    return None


def get_subject_neg(token):
    for child in token.children:
        if child.dep == det and child.text.lower() == "no":
            return child

    return None


def get_object_neg(token):
    for child in token.children:
        if child.dep == neg:
            return child

    return None


def get_object_adj(token):
    for child in token.children:
        if child.dep == amod and child.pos == ADJ:
            return child
        if child.dep == advmod and child.pos == ADV:
            return child

    return None
