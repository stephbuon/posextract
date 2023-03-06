from dataclasses import dataclass

from spacy.matcher import DependencyMatcher
from spacy.tokens import *
from spacy.symbols import *


@dataclass
class VerbPhrase:
    first: Token
    second: Token

    def __hash__(self):
        return hash(self.first) + hash(self.second)

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
        raise NotImplementedError

    @property
    def object_search_root(self):
        raise NotImplementedError

    @property
    def head(self):
        raise NotImplementedError

    @property
    def pos(self):
        return VERB

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
    def subject_search_root(self):
        return self.first

    @property
    def object_search_root(self):
        return self.second

    @property
    def head(self):
        return self.first.head

    @property
    def text(self):
        return self.second.text

    @property
    def lemma_(self):
        return self.second.lemma_


class ConjVerbPhrase(VerbPhrase):
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
    def text(self):
        return self.second.text

    @property
    def lemma_(self):
        return self.second.lemma_


class CCompVerbPhrase(VerbPhrase):
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
    def text(self):
        return self.first.text

    @property
    def lemma_(self):
        return self.first.lemma_


class XCompVerbPhrase(CCompVerbPhrase):
    def _get_part(self, token):
        part = ' '

        try:
            child = next(token.children)
            if child.pos == PART and child.dep == aux:
                part = f' {child.text} '
        except StopIteration:
            pass

        return part

    @property
    def text(self):
        part = self._get_part(self.second)
        return f'{self.first.text}{part}{self.second.text}'

    @property
    def lemma_(self):
        part = self._get_part(self.second)
        return f'{self.first.lemma_}{part}{self.second.lemma_}'


VERB_PHRASE_TABLE = {
    'advcl-verb-phrase': ADVCLVerbPhrase,
    'conj-verb-phrase': ConjVerbPhrase,
    'ccomp-verb-phrase': CCompVerbPhrase,
    'xcomp-verb-phrase': XCompVerbPhrase,
}


def add_verb_phrase_patterns(matcher: DependencyMatcher):
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

    matcher.add("ccomp-verb-phrase", [
        [
            # anchor token: verb
            {
                "RIGHT_ID": "verb",
                "RIGHT_ATTRS": {"POS": "VERB"}
            },

            # verb -> verb2
            {
                "LEFT_ID": "verb",
                "REL_OP": ">",
                "RIGHT_ID": "verb2",
                "RIGHT_ATTRS": {"DEP": "ccomp", "POS": "VERB"}
            },
        ],

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
                "RIGHT_ATTRS": {"DEP": "ccomp", "POS": "AUX"}
            },
        ],
    ])

    matcher.add("xcomp-verb-phrase", [
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
                "RIGHT_ATTRS": {"DEP": "xcomp", "POS": "AUX"}
            },
        ],
    ])
