from . import rules
import argparse
import os
import spacy
from spacy.tokens import Doc, Token
from posextractor.util import subject_search, object_search, is_verb
import pandas as pd

rule_funcs = [
    rules.rule1,
    rules.rule2,
    rules.rule3,
    rules.rule4,
    rules.rule5,
    rules.rule6,
    rules.rule7,
    rules.rule8,
    rules.rule9,
    rules.rule10,
    rules.rule11,
    rules.rule12,
]

nlp = spacy.load("en_core_web_sm")


def graph_tokens(doc: Doc, verbose=False, metadata=None):
    verbs = []

    for token in doc:  # type: Token
        if is_verb(token):
            verbs.append(token)

    extractions = []

    for verb in verbs:
        if verbose: print('beginning triple search for verb:', verb)

        # Search for the subject.
        subjects = subject_search(verb)

        # Search for the objects.
        objects = object_search(verb)

        if not subjects:
            if verbose: print('Couldnt find subjects.')
            continue

        if not objects:
            if verbose: print('Couldnt find objects.')
            continue

        for subject in subjects:
            for object_pair in objects:
                poa, obj = object_pair
                if verbose: print('\tpossible triple:', subject.lemma_, verb.lemma_, poa if poa else '', obj.lemma_)

                for rule in rule_funcs:
                    if rule(verb, subject, obj, poa):
                        if verbose: print('\tmatched with', rule.__name__)
                        if metadata:
                            extractions.append((verb, subject, obj, poa, metadata))
                        else:
                            extractions.append((verb, subject, obj, poa))
                        break
                else:
                    if verbose: print('\tNo matching rule found.')

    return extractions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='posextractor')
    parser.add_argument('input', metavar='input', type=str,
                        help='a filepath to a csv file or an input string')
    parser.add_argument('output', metavar='output', type=str,
                        help='an output path')
    parser.add_argument('--data_column', type=str, default=None, metavar='data_col',
                        help='what column to use if a csv is given', dest='data_column')
    parser.add_argument('--id_column', type=str, default=None, metavar='id_col',
                        help='what column to use if a csv is given', dest='id_column')
    args = parser.parse_args()
    is_file = os.path.isfile(args.input)

    inputs = []
    outputs = []

    if is_file:
        if args.data_column is None or args.id_column is None:
            exit('Must specify column name for data')
        df = pd.read_csv(args.input, index_col=args.id_column, usecols=[args.data_column, args.id_column])

        for i, row in df.iterrows():
            doc = nlp(row[args.data_column])
            extractions = graph_tokens(doc, metadata=i)
            outputs.extend(extractions)
    else:
        doc = nlp(args.input)
        extractions = graph_tokens(doc, metadata=None)
        outputs.extend(extractions)

    out_columns = ['verb', 'subject', 'obj', 'poa']
    if is_file:
        out_columns.append(args.id_column)
    output_df = pd.DataFrame(outputs, columns=out_columns)
    if is_file:
        output_df.set_index(args.id_column, inplace=True)
    output_df.to_csv(args.output)
    print('Number of extractions: %d' % len(outputs))
