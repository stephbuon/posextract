import collections
from typing import List, Union, Iterable

import pandas

from . import rules
import argparse
import os
from spacy.tokens import Doc
from spacy.symbols import aux
from .util import *
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


def visit_verb(verb, parent_subjects, parent_objects, verbose=False):
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
                    yield extraction
                    break
            else:
                if verbose: print('\tNo matching rule found.\n')

    yield from visit_token(verb, parent_subjects=subjects, verbose=verbose)


def visit_token(token, parent_subjects, verbose=False):
    for child in token.children:
        if is_verb(child):
            yield from visit_verb(child, parent_subjects, parent_objects=[], verbose=verbose)
        else:
            # Reset inherited subjects and objects.
            yield from visit_token(child, [], verbose=verbose)


def graph_tokens(doc: Doc, verbose=False) -> List[TripleExtraction]:
    root_verb = None

    for token in doc:
        if is_root(token):
            root_verb = token
            if verbose: print(f"Root verb is {root_verb}")
            break

    if root_verb is None:
        if verbose: print('Could not find root verb.')
        return []

    extraction_set = set()
    triple_extractions = list(visit_verb(root_verb, [], [], verbose=verbose))
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
            ext_main = next(
                ext for ext in dupe_list if ext.object.dep == pobj or ext.object.dep == dobj or ext.object.dep == acomp)
            adjectives = []

            for ext in dupe_list:
                if ext.object.i == ext_main.object.i:
                    continue
                if ext.object.dep == advmod and not ext.poa:
                    adjectives.append(ext.object)
                else:
                    new_extractions.append(ext)

            ext_main.adjectives = adjectives
            new_extractions.append(ext_main)

        except StopIteration:
            # No extraction can be combined
            new_extractions.append(dupe_list[0])
            continue

    return new_extractions


def extract(input_object: Union[str, Iterable[str]], combine_adj: bool = False, lemmatize: bool = False,
            add_aux: bool = False, verbose: bool = False,
            want_dataframe: bool = False) -> Union[List[TripleExtractionFlattened], pandas.DataFrame]:
    output_extractions = []

    if type(input_object) == str:
        input_object = [input_object, ]
    elif not isinstance(input_object, collections.Iterable):
        raise ValueError('extract_triples: input should be a string or a collection of strings')

    for i, doc in enumerate(input_object):
        doc = nlp(doc)
        extractions = graph_tokens(doc, verbose=verbose)
        output_extractions.extend(extractions)

    if combine_adj:
        if verbose: print('Combining triples...')
        output_extractions = post_process_combine_adj(output_extractions)

    if add_aux:
        for triple in output_extractions:
            for child in triple.verb.children:
                if child.dep == aux and child.text == 'will':
                    triple.aux_verb = child
                    break

    output_extractions = [triple.flatten(lemmatize=lemmatize) for triple in output_extractions]

    if want_dataframe:
        extractions_df = pd.DataFrame([t.__dict__ for t in output_extractions])
        return extractions_df

    return output_extractions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='posextractor')
    parser.add_argument('input', metavar='input', type=str,
                        help='a filepath to a csv file or an input string')
    parser.add_argument('output', metavar='output', type=str,
                        help='an output path')
    parser.add_argument('--data-column', type=str, default=None, metavar='data_col',
                        help='what column to use if a csv is given', dest='data_column')
    parser.add_argument('--id-column', type=str, default=None, metavar='id_col',
                        help='what column to use if a csv is given', dest='id_column')
    parser.add_argument('--file-delimiter', default='comma', const='comma', nargs='?',
                        choices=['comma', 'pipe', 'tab'],
                        help='delimiter character for data file (default: %(default)s)')
    parser.add_argument('--post-combine-adj', action='store_true')
    parser.add_argument('--lemma', action='store_true')
    parser.add_argument('--add-auxiliary', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    is_file = os.path.isfile(args.input)

    inputs = []
    outputs = []

    delimiter = {'comma': ',', 'pipe': '|', 'tab': '\t'}[args.file_delimiter]

    input_values = None
    df = None

    if is_file:
        if args.verbose:
            print('Loading input (%s) as a CSV file...' % args.input)
            print('delimiter:', args.file_delimiter)
        if args.data_column is None:
            exit('Invalid arguments: Must specify column name for data using --data-column')

        usecols = [args.data_column, ]

        if args.id_column is not None:
            usecols.append(args.id_column)

        df = pd.read_csv(args.input, index_col=args.id_column, usecols=usecols, delimiter=delimiter)
        input_values = df[args.data_column]
    else:
        input_values = [args.input, ]

    with open(args.output, 'w+') as f:
        pass

    extraction_count = 0
    header = True

    for i, data_str in enumerate(input_values):
        triples_df = extract(data_str, combine_adj=args.post_combine_adj, lemmatize=args.lemma,
                             add_aux=args.add_auxiliary, verbose=args.verbose, want_dataframe=True)
        extraction_count += len(triples_df)
        if df is not None:
            triples_df['sentence_id'] = df.index[i]
        triples_df.to_csv(args.output, mode='a', sep=delimiter, header=header, index=False)

        if header:
            header = False

    if args.verbose:
        print('Number of extractions: %d' % extraction_count)

__all__ = ['extract', ]
