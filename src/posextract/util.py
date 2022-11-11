from spacy.symbols import *
from spacy.tokens import *
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
    object_adjectives: str = ''
    object: str = ''
    object_prep: str = ''
    object_prep_noun: str = ''

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
    object_adjectives: Optional[List[Token]] = None
    object: Optional[Token] = None
    object_prep: Optional[Token] = None
    object_prep_noun: Optional[Token] = None

    def flatten(self, lemmatize=False, compound_subject=True, compound_object=True) -> TripleExtractionFlattened:
        kwargs = {k: v for k, v in self.__dict__.items() if v is not None}

        if lemmatize:
            if self.object:
                kwargs['object'] = self.object.lemma_
            if self.verb:
                kwargs['verb'] = self.verb.lemma_
            if self.subject:
                kwargs['subject'] = self.subject.lemma_
        else:
            if (self.verb and self.subject) and (self.verb.i < self.subject.i):
                kwargs['verb'] = self.verb.lemma_

        if self.object_adjectives:
            kwargs['object_adjectives'] = ' '.join((adj.text for adj in self.object_adjectives))

        for k, v in kwargs.items():
            if type(v) != str:
                kwargs[k] = str(v)

        if compound_subject:
            for child in self.subject.children:
                if child.dep_ == "compound":
                    kwargs['subject'] = child.text + ' ' + kwargs['subject']

        if self.object.dep == advmod and self.object.pos == ADV:
            if self.object.head.pos == ADJ:
                kwargs['object'] += ' ' + self.object.head.text

        if compound_object:
            for child in reversed(list(self.object.children)):
                if child.dep_ == "compound":
                    kwargs['object'] = child.text + ' ' + kwargs['object']

        for verb_child in self.verb.children:
            if verb_child.pos == ADP and verb_child.dep == prt:
                kwargs['verb'] += ' ' + verb_child.text

        return TripleExtractionFlattened(
            **kwargs
        )

    def get_triple_hash(self) -> int:
        default_empty_str = lambda x: x.text.lower() if x else ''
        return hash((default_empty_str(self.subject), default_empty_str(self.verb), default_empty_str(self.object)))


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


def subject_search(token: Token, verbose=False):
    objects = []

    visited = set()
    considering = [token, ]

    if verbose:
        print('\tDoing subject search for token: ', token)
        print('\tverb.head', token.head)
        print('\tverb.children', list(token.children))

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

                if verbose:
                    print('\t\t(verb=%s) considering child:' % token, child.text, 'with POS=', child.pos_)
                    print('\t\tdependency of %s->%s:' % (candidate, child), child.dep_)
                considering.append(child)

        parent = candidate.head
        if parent not in visited:
            if (parent.pos == VERB or parent.pos == AUX) and (candidate.dep == conj or candidate.dep == advcl):
                continue

            print('\t\t(verb=%s) considering parent:' % token, parent.text, 'with POS=', parent.pos_)
            print('\t\tdependency of %s->%s:' % (parent, candidate), candidate.dep_)
            considering.append(parent)

    return objects


def get_verb_neg(token, up=True):
    for child in token.children:
        if child.dep == neg:
            return child

    # if up and token.head.pos == VERB:
    #     parent_negation = get_verb_neg(token.head, up=False)
    #     if parent_negation:
    #         return parent_negation

    return None


def get_subject_neg(token):
    for child in token.children:
        if child.dep == det and child.text.lower() in ("no", "not", "never"):
            return child

    return None


def get_object_neg(token):
    for child in token.children:
        if child.dep == det and child.text.lower() in ("no", "not", "never"):
            return child

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
