from pathlib import Path

import spacy
from spacy import displacy
from spacy.symbols import *
from spacy.tokens import Doc, Token
from posextractor.util import subject_search, object_search, is_verb, is_object

import posextractor.rules

rule_funcs = [
    posextractor.rules.rule1,
    posextractor.rules.rule2,
    posextractor.rules.rule3,
    posextractor.rules.rule4,
    posextractor.rules.rule5,
    posextractor.rules.rule6,
    posextractor.rules.rule7,
    posextractor.rules.rule8,
    posextractor.rules.rule9,
    posextractor.rules.rule10,
    posextractor.rules.rule11,
    posextractor.rules.rule12,
]

nlp = spacy.load("en_core_web_sm")


def graph_tokens(doc: Doc):
    verbs = []

    for token in doc:  # type: Token
        if is_verb(token):
            verbs.append(token)

    for verb in verbs:
        print('beginning triple search for verb:', verb)

        # Search for the subject.
        subjects = subject_search(verb)

        # Search for the objects.
        objects = object_search(verb)

        if not subjects:
            print('Couldnt find subjects.')
            continue

        if not objects:
            print('Couldnt find objects.')
            continue

        for subject in subjects:
            for object_pair in objects:
                poa, obj = object_pair
                print('\tpossible triple:', subject.lemma_, verb.lemma_, poa if poa else '', obj.lemma_)

                for rule in rule_funcs:
                    if rule(verb, subject, obj, poa):
                        print('\tmatched with', rule.__name__)
                        break
                else:
                    print('\tNo matching rule found.')

                print('\n')

        print('\n\n')


# s = "The establishment of the Mauritius garrison has been slightly reduced during the last few years, though the actual strength was somewhat depleted at the beginning of that period, owing to the war."

s = "Mr. Goulding begged to trouble the House for a few minutes, as he was possessed of some local knowledge on the subject of the petition from John Knight."
doc = nlp(s)

graph_tokens(doc)
svg = displacy.render(doc, style="dep", jupyter=False)
file_name = "displacy.svg"
output_path = Path(f'./{file_name}')
output_path.open("w", encoding="utf-8").write(svg)
