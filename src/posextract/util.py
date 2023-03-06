from dataclasses import dataclass
from typing import NamedTuple, Union

import spacy.tokens
from spacy.matcher import DependencyMatcher
from spacy.symbols import *
from spacy.tokens import *

__DEP_MATCHER = None


@dataclass
class VerbPhrase:
    first: Token
    second: Token

    def __hash__(self):
        return hash(self.first) + hash(self.second)

    @property
    def dep(self):
        raise NotImplementedError

    @property
    def dep_(self):
        raise NotImplementedError

    @property
    def children(self):
        raise NotImplementedError

    @property
    def subject_search_root(self):
        raise NotImplementedError

    @property
    def object_search_root(self):
        raise NotImplementedError

    @property
    def head(self):
        raise NotImplementedError

    @property
    def pos(self):
        raise NotImplementedError

    @property
    def text(self):
        raise NotImplementedError

    @property
    def lemma_(self):
        raise NotImplementedError

    def __contains__(self, item):
        return item == self.first or item == self.second

    def __str__(self):
        return self.text

    def __eq__(self, other):
        if isinstance(other, VerbPhrase):
            return self.first == other.first and self.second == other.second
        else:
            return other == self.first or other == self.second


class ADVCLVerbPhrase(VerbPhrase):
    @property
    def dep(self):
        return self.first.dep

    @property
    def dep_(self):
        return self.first.dep_

    @property
    def children(self):
        yield from self.first.children
        yield from self.second.children

    @property
    def subject_search_root(self):
        return self.first

    @property
    def object_search_root(self):
        return self.second

    @property
    def head(self):
        return self.first.head

    @property
    def pos(self):
        return VERB

    @property
    def text(self):
        return self.second.text

    @property
    def lemma_(self):
        return self.second.lemma_


class ConjVerbPhrase(VerbPhrase):
    @property
    def dep(self):
        return self.first.dep

    @property
    def dep_(self):
        return self.first.dep_

    @property
    def children(self):
        yield from self.first.children
        yield from self.second.children

    @property
    def subject_search_root(self):
        return self.first

    @property
    def object_search_root(self):
        return self.second

    @property
    def head(self):
        return self.first.head

    @property
    def pos(self):
        return VERB

    @property
    def text(self):
        return self.second.text

    @property
    def lemma_(self):
        return self.second.lemma_


def get_dep_matcher():
    global __DEP_MATCHER

    if __DEP_MATCHER is None:
        matcher = DependencyMatcher(get_nlp().vocab)

        matcher.add("advcl-verb-phrase", [
            [
                # anchor token: aux_verb
                {
                    "RIGHT_ID": "aux_verb",
                    "RIGHT_ATTRS": {"POS": "AUX"}
                },

                # aux_verb -> verb
                {
                    "LEFT_ID": "aux_verb",
                    "REL_OP": ">",
                    "RIGHT_ID": "verb",
                    "RIGHT_ATTRS": {"DEP": "advcl", "POS": "VERB"}
                },
            ],
            [
                # anchor token: verb1
                {
                    "RIGHT_ID": "verb1",
                    "RIGHT_ATTRS": {"POS": "VERB"}
                },

                # verb1 -> verb2
                {
                    "LEFT_ID": "verb1",
                    "REL_OP": ">",
                    "RIGHT_ID": "verb2",
                    "RIGHT_ATTRS": {"DEP": "advcl", "POS": "VERB"}
                },
            ]
        ])

        matcher.add("conj-verb-phrase", [
            [
                # anchor token: verb
                {
                    "RIGHT_ID": "verb",
                    "RIGHT_ATTRS": {"POS": "VERB"}
                },

                # verb -> aux_verb
                {
                    "LEFT_ID": "verb",
                    "REL_OP": ">",
                    "RIGHT_ID": "aux_verb",
                    "RIGHT_ATTRS": {"DEP": "conj", "POS": "AUX"}
                },
            ],

            [
                # anchor token: aux_verb
                {
                    "RIGHT_ID": "aux_verb",
                    "RIGHT_ATTRS": {"POS": "AUX"}
                },

                # aux_verb -> verb
                {
                    "LEFT_ID": "aux_verb",
                    "REL_OP": ">",
                    "RIGHT_ID": "verb",
                    "RIGHT_ATTRS": {"DEP": "conj", "POS": "VERB"}
                },
            ],
            [
                # anchor token: verb
                {
                    "RIGHT_ID": "verb",
                    "RIGHT_ATTRS": {"POS": "VERB"}
                },

                # verb -> verb2
                {
                    "LEFT_ID": "verb",
                    "REL_OP": ">>",
                    "RIGHT_ID": "verb2",
                    "RIGHT_ATTRS": {"DEP": "conj", "POS": "VERB"}
                },
            ],

        ])

        __DEP_MATCHER = matcher

    return __DEP_MATCHER


VERB_PHRASE_TABLE = {
    'advcl-verb-phrase': ADVCLVerbPhrase,
    'conj-verb-phrase': ConjVerbPhrase,
}


def should_consider_verb_phrase(verb_phrase: VerbPhrase):
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

__NLP = None


def get_nlp():
    global __NLP
    if __NLP is None:
        __NLP = spacy.load("en_core_web_sm")
    return __NLP


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
