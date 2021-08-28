import spacy
from spacy.symbols import *
from spacy.tokens import Doc, Token
from posextractor.util import subject_search, object_search

import posextractor.rules

rule_funcs = [
    posextractor.rules.rule1,
    posextractor.rules.rule2,
    posextractor.rules.rule3,
    posextractor.rules.rule4,
]

nlp = spacy.load("en_core_web_sm")


def graph_tokens(doc: Doc):
    verbs = []

    for token in doc:  # type: Token
        if token.pos == VERB:
            verbs.append(token)

    for verb in verbs:
        print('beginning triple search for verb:', verb)

        # Search for the subject.
        subjects = subject_search(verb)

        # Search for the objects.
        objects = object_search(verb)

        for subject in subjects:
            for object_pair in objects:
                poa, obj = object_pair
                print('\tpossible triple:', subject, verb, poa if poa else '', obj)

                for rule in rule_funcs:
                    if rule(verb, subject, obj, poa):
                        print('\tmatched with', rule.__name__)
                        break
                else:
                    print('\tNo matching rule found.')

                print('\n')

        print('\n\n')


s = "Taking all these circumstances into consideration, he must again say that he did not think the returning officer would be able to look over more than 120 ballot papers an hour, and in the case of a large constituency the result of the poll would not be known for a fortnight"
doc = nlp(s)
# displacy.serve(doc, style="dep")
graph_tokens(doc)

