from typing import List

from posextract import rules
import argparse
import os
from spacy.tokens import Doc
from posextract.util import *
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
        print('verb dep=', verb.dep_)
        print('\tparent_subjects=', parent_subjects)
        print('\tparent_objects=', parent_objects)

    # Search for the subject.
    subjects = subject_search(verb) + parent_subjects

    # Search for the objects.
    objects = object_search(verb) + parent_objects

    # Remove duplicates.
    subjects = list(set(subjects))
    objects = list(set(objects))

    if verbose:
        print('\tsubjects=', subjects)
        print('\tobjects=', objects)

    if not subjects:
        if verbose: print('Could not find subjects.')

    if not objects:
        if verbose: print('Could not find objects.')

    neg_adverb = get_verb_neg(verb)

    for subject_negdat, subject in subjects:
        for poa, obj_negdat, obj in objects:
            if verbose: print('\tconsidering triple:', subject, verb, poa if poa else '', obj)

            for rule in rule_funcs:
                if rule(verb, subject, obj, poa):
                    if verbose: print('\tmatched with', rule.__name__, '\n')

                    extraction = TripleExtraction(
                            subject_negdat=subject_negdat, subject=subject,
                            neg_adverb=neg_adverb, verb=verb,
                            poa=poa, object_negdat=obj_negdat, object=obj)

                    if metadata:
                        yield extraction, metadata
                    else:
                        yield extraction
                    break
            else:
                if verbose: print('\tNo matching rule found.\n')

    for child in verb.children:
        if is_verb(child):
            yield from visit_verb(child, subjects, [], metadata, verbose=verbose)
        else:
            # Reset inherited subjects and objects.
            yield from visit_token(child, [], [], metadata, verbose=verbose)


def visit_token(token, parent_subjects, parent_objects, metadata, verbose=False):
    for child in token.children:
        if is_verb(child):
            yield from visit_verb(child, parent_subjects, [], metadata, verbose=verbose)
        else:
            yield from visit_token(child, [], [], metadata, verbose=verbose)


def graph_tokens(doc: Doc, verbose=False, metadata=None):
    root_verb = None

    for token in doc:
        if is_root(token):
            root_verb = token
            print(f"Root verb is {root_verb}")
            break

    if root_verb is None:
        print('Could not find root verb.')
        return []

    extraction_set = set()
    triple_extractions = list(visit_verb(root_verb, [], [], metadata, verbose=verbose))
    triple_extractions_no_duplicates = []

    for triple in triple_extractions:
        if str(triple) not in extraction_set:
            triple_extractions_no_duplicates.append(triple)
            extraction_set.add(str(triple))

    return triple_extractions


def post_process_combine_adj(extractions: List[TripleExtraction]):
    possible_dupes = {}

    for extraction in extractions:
        key = (extraction.subject.i, extraction.verb.i)
        possible_dupes.setdefault(key, []).append(extraction)

    new_extractions = []

    for key, dupe_list in possible_dupes.items():

        if len(dupe_list) == 1:
            new_extractions.append(dupe_list[0])
            continue

        # Find the extraction with a pobj or dobj
        try:
            ext_main = next(ext for ext in dupe_list if ext.object.dep == pobj or ext.object.dep == dobj or ext.object.dep == acomp)
            adjectives = []

            for ext in dupe_list:
                if ext.object.i == ext_main.object.i:
                    continue
                if ext.object.dep == advmod and not ext.poa:
                    adjectives.append(str(ext.object))
                else:
                    new_extractions.append(ext)

            adjectives = ' '.join(adjectives)

            ext_main = TripleExtraction(
                subject_negdat=ext_main.subject_negdat,
                subject=ext_main.subject,
                neg_adverb=ext_main.neg_adverb,
                verb=ext_main.verb,
                poa=ext_main.poa,
                object_negdat=ext_main.object_negdat,
                adjectives=adjectives,
                object=ext_main.object,
            )

            new_extractions.append(ext_main)

        except StopIteration:
            # No extraction can be combined
            new_extractions.append(dupe_list[0])
            continue

    return new_extractions


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
    parser.add_argument('--post-combine-adj', action='store_true')
    parser.add_argument('--lemma', action='store_true')
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
            extractions = graph_tokens(doc, metadata=i, verbose=True)
            outputs.extend(extractions)
    else:
        doc = nlp(args.input)
        extractions = graph_tokens(doc, metadata=None)
        outputs.extend(extractions)

    if args.post_combine_adj:
        print('Combining triples...')
        outputs = post_process_combine_adj(outputs)\

    if args.lemma:
        outputs = [triple.lemmatized() for triple in outputs]

    out_columns = ['subject_negdat', 'subject', 'neg_adverb', 'verb', 'poa', 'object_negdat', 'adjectives', 'object']
    if is_file:
        out_columns.append(args.id_column)
    output_df = pd.DataFrame(outputs, columns=out_columns)
    if is_file:
        output_df.set_index(args.id_column, inplace=True)
    output_df.to_csv(args.output)
    print('Number of extractions: %d' % len(outputs))