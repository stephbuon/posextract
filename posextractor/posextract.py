from . import rules
import argparse
import os
import spacy
from spacy.tokens import Doc, Token
from posextractor.util import subject_search, object_search, is_root, is_verb
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


def visit_verb(verb, parent_subjects, parent_objects, metadata, verbose=False):
    if verbose:
        print('beginning triple search for verb:', verb)
        print('\tparent_subjects=', parent_subjects)
        print('\tparent_objects=', parent_objects)

    # Search for the subject.
    subjects = subject_search(verb) + parent_subjects

    # Search for the objects.
    objects = object_search(verb) + parent_objects

    # Remove duplicates.
    # TODO: Not sure if this is needed.
    # subjects = list(set(subjects))
    # objects = list(set(objects))

    if not subjects:
        if verbose: print('Could not find subjects.')

    if not objects:
        if verbose: print('Could not find objects.')

    for subject in subjects:
        for object_pair in objects:
            poa, obj = object_pair
            if verbose: print('\tconsidering triple:', subject.lemma_, verb.lemma_, poa if poa else '', obj.lemma_)

            for rule in rule_funcs:
                if rule(verb, subject, obj, poa):
                    if verbose: print('\tmatched with', rule.__name__, '\n')
                    if metadata:
                        yield verb, subject, obj, poa, metadata
                    else:
                        yield verb, subject, obj, poa
                    break
            else:
                if verbose: print('\tNo matching rule found.\n')

    for child in verb.children:
        if is_verb(child):
            yield from visit_verb(child, subjects, objects, metadata, verbose=verbose)


def graph_tokens(doc: Doc, verbose=False, metadata=None):
    root_verb = None

    for token in doc:
        if is_root(token):
            root_verb = token
            break

    if root_verb is None:
        print('Could not find root verb.')
        return []

    triple_extractions = list(visit_verb(root_verb, [], [], metadata, verbose=verbose))

    return triple_extractions


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
