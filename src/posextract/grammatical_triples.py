import collections
import copy
from typing import List, Union, Iterable

import pandas

import argparse
import os

from posextract.traversal import graph_tokens
from posextract.triple_extraction import TripleExtraction, TripleExtractionFlattened
from posextract.util import *
import pandas as pd

nlp = get_nlp()


def post_process_combine_adj(extractions: List[TripleExtraction]):
    possible_dupes = {}

    for extraction in extractions:
        if isinstance(extraction.verb, VerbPhrase):
            continue
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

            ext_main.object_adjectives = adjectives
            new_extractions.append(ext_main)

        except StopIteration:
            # No extraction can be combined
            new_extractions.append(dupe_list[0])
            continue

    return new_extractions


def post_process_prep_phrase(extraction: TripleExtraction):
    # Rule 1: check if the prep "of" or "to" is the child of the object, and if it is, check to see if a noun is the child of the preposition.
    # Rule 2: check if the prep "with" is the child of the verb, and it if is, see if the pobj is the child of the preposition.

    for child in extraction.object.children:
        if child.text in ('of', 'to'):
            nouns = [childchild for childchild in child.children if childchild.pos == NOUN or childchild.dep == pobj]

            if len(nouns) != 1:
                continue

            extraction.object_prep = child
            extraction.object_prep_noun = nouns[0]

            return extraction

    # for child in extraction.verb.children:
    #     if child == extraction.poa:
    #         continue
    #     if child.text == 'with':
    #         pobjs = [childchild for childchild in child.children if childchild.dep == pobj]
    #
    #         if len(pobjs) != 1:
    #             continue
    #
    #         extraction.object_prep = child
    #         extraction.object_prep_noun = pobjs[0]
    #         return extraction

    return extraction


def post_process_conj_triples(triple: TripleExtraction):
    new_extractions = []

    # Look for additional triples due to conj dependency

    visited = set()
    considering = list(triple.subject.children)

    while considering:
        token = considering.pop(-1)
        if token in visited:
            continue
        visited.add(token)
        if token.pos == NOUN and token.dep == conj:
            new_triple = copy.copy(triple)
            new_triple.subject = token
            new_extractions.append(new_triple)
            considering.extend(token.children)

    visited = set()
    considering = list(triple.object.children)

    while considering:
        token = considering.pop(-1)
        if token in visited:
            continue
        visited.add(token)
        if token.pos == NOUN and token.dep == conj:
            new_triple = copy.copy(triple)
            new_triple.object = token
            new_extractions.append(new_triple)
            considering.extend(token.children)

    return new_extractions


def post_process_adj_acomp(triple: TripleExtraction):
    if triple.object.pos != ADJ or triple.object.dep != acomp:
        return []

    new_triples = []
    visited = set()
    considering = list(triple.object.children)

    while considering:
        candidate = considering.pop(-1)

        if candidate in visited:
            continue

        visited.add(candidate)

        if candidate.pos == ADJ and candidate.dep == conj:
            new_triple = copy.copy(triple)
            new_triple.object = candidate
            new_triples.append(new_triple)

        for child in candidate.children:
            if child not in visited:
                if child.pos != ADJ:
                    continue
                considering.append(child)

    return new_triples


def resolve_coreferences(triple: TripleExtraction):
    if triple.subject.text.lower() == 'which':
        if triple.subject.head.pos == NOUN:
            triple.subject = triple.subject.head

    if triple.subject.text.lower() == 'who' and triple.subject.pos == PRON:
        if triple.verb == triple.subject.head:
            noun = triple.verb.head
            if noun.pos in (NOUN, PROPN) and triple.verb.dep == relcl:
                triple.subject = noun


def add_auxiliary_verb(triple: TripleExtraction):
    for child in triple.verb.children:
        if child.dep == aux:
            triple.aux_verb = child
            break


def yield_non_duplicate_triples(extractions: List[TripleExtraction]) -> List[TripleExtraction]:
    hashes = set()
    for triple in extractions:
        h = triple.get_triple_hash()
        if h not in hashes:
            yield triple
            hashes.add(h)


def extract_one(doc: Doc, extractor_options: TripleExtractorOptions = None,
                verbose: bool = False, flatten: bool = False):
    if extractor_options is None:
        extractor_options = TripleExtractorOptions()

    extractions = graph_tokens(doc, verbose=verbose)
    extractions = list(yield_non_duplicate_triples(extractions))

    for triple in extractions:
        extractions.extend(post_process_conj_triples(triple))
        extractions.extend(post_process_adj_acomp(triple))

    if extractor_options.combine_adj:
        extractions = post_process_combine_adj(extractions)

    extractions = list(yield_non_duplicate_triples(extractions))

    for triple in extractions:
        resolve_coreferences(triple)

        if extractor_options.add_auxiliary:
            add_auxiliary_verb(triple)

        if extractor_options.prep_phrase:
            post_process_prep_phrase(triple)

    if flatten:
        extractions = [
            triple.flatten(lemmatize=extractor_options.lemmatize,
                           compound_subject=extractor_options.compound_subject,
                           compound_object=extractor_options.compound_object)
            for triple in extractions]

    return extractions


def extract(input_object: Union[str, Iterable[str]], extractor_options: TripleExtractorOptions = None,
            verbose: bool = False,
            want_dataframe: bool = False) -> Union[List[TripleExtractionFlattened], pandas.DataFrame]:
    if extractor_options is None:
        extractor_options = TripleExtractorOptions()

    output_extractions = []

    if type(input_object) == str:
        input_object = [input_object, ]
    elif not isinstance(input_object, collections.Iterable):
        raise ValueError('extract_triples: input should be a string or a collection of strings')

    for i, doc in enumerate(input_object):
        doc = nlp(doc)
        output_extractions.extend(
            extract_one(doc,extractor_options, flatten=True, verbose=verbose))

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
    parser.add_argument('--prep-phrase', action='store_true')
    parser.add_argument('--no-compound-subject', action='store_true')
    parser.add_argument('--no-compound-object', action='store_true')

    args = parser.parse_args()
    is_file = os.path.isfile(args.input)

    extractor_options = TripleExtractorOptions(
        compound_subject=not args.no_compound_subject,
        compound_object=not args.no_compound_object,
        combine_adj=args.post_combine_adj,
        add_auxiliary=args.add_auxiliary,
        prep_phrase=args.prep_phrase,
        lemmatize=args.lemma
    )

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
        triples_df = extract(data_str, extractor_options=extractor_options, verbose=args.verbose, want_dataframe=True)
        extraction_count += len(triples_df)
        if df is not None:
            triples_df['sentence_id'] = df.index[i]
        triples_df.to_csv(args.output, mode='a', sep=delimiter, header=header, index=False)

        if header:
            header = False

    if args.verbose:
        print('Number of extractions: %d' % extraction_count)

__all__ = ['extract', 'extract_one']
