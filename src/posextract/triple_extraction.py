from typing import Optional, List, Union

from spacy.matcher import DependencyMatcher
from spacy.symbols import *
from spacy.tokens import *
from dataclasses import dataclass

from posextract.util import VerbPhrase


@dataclass
class TripleExtractionFlattened:
    subject_negdet: str = ''
    subject: str = ''
    neg_adverb: str = ''
    neg_adverb_part: str = ''
    aux_verb: str = ''
    verb: str = ''
    poa_neg: str = ''
    poa: str = ''
    object_negdet: str = ''
    object_adjectives: str = ''
    object: str = ''
    object_prep: str = ''
    object_prep_noun: str = ''
    rule: str = ''

    def astuple(self):
        return (v for k, v in self.__dict__.items() if k != 'rule')

    def __str__(self):
        return ' '.join((str(v) for v in self.astuple() if v))


EMPHASIS_ADJ_LIST = ('very', 'much', 'most', 'utterly', 'as')


@dataclass
class TripleExtraction:
    subject_negdet: Optional[Token] = None
    subject: Optional[Token] = None
    neg_adverb: Optional[Token] = None
    neg_adverb_part: Optional[Token] = None
    aux_verb: Optional[Token] = None
    verb: Optional[Union[Token, VerbPhrase]] = None
    poa_neg: Optional[Token] = None
    poa: Optional[Token] = None
    object_negdet: Optional[Token] = None
    object_adjectives: Optional[List[Token]] = None
    object: Optional[Token] = None
    object_prep: Optional[Token] = None
    object_prep_noun: Optional[Token] = None
    rule: str = ''
    verb_phrase: bool = False

    def flatten(self, lemmatize=False, compound_subject=True, compound_object=True) -> TripleExtractionFlattened:
        kwargs = {k: v for k, v in self.__dict__.items() if v is not None}

        del kwargs['verb_phrase']

        if lemmatize:
            if self.object:
                kwargs['object'] = self.object.lemma_
            if self.verb:
                kwargs['verb'] = self.verb.lemma_
            if self.subject:
                kwargs['subject'] = self.subject.lemma_
        else:
            if hasattr(self.verb, 'i') and (self.verb and self.subject) and (self.verb.i < self.subject.i):
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
            if self.object.head.pos == ADJ and self.object.text.lower() in EMPHASIS_ADJ_LIST:
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
