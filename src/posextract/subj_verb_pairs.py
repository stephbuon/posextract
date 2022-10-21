import argparse
import os
import warnings;

from .util import get_verb_neg

warnings.simplefilter('ignore')
import pandas as pd
import spacy
from spacy.symbols import nsubj, nsubjpass, VERB
import collections
from typing import List


nlp = spacy.load('en_core_web_sm', disable=['ner'])


from typing import NamedTuple


def extract_old(hansard, col, **kwargs):
    kw = kwargs.get('keep', None)

    if kw == 'keep':
        hansard['parsed_text'] = [doc for doc in nlp.pipe(hansard[col].tolist())]  # this turns into env
        hansard['subj_verb_pair'] = hansard['parsed_text'].apply(rule)
        hansard = hansard.loc[:, hansard.columns != 'parsed_text']
    else:
        hansard[col] = [doc for doc in nlp.pipe(hansard[col].tolist())]
        hansard['subj_verb_pair'] = hansard[col].apply(rule)
        hansard = hansard.loc[:, hansard.columns == 'subj_verb_pair']

    hansard = hansard[hansard.astype(str)['subj_verb_pair'] != '[]']

    hansard = hansard.explode('subj_verb_pair')

    return hansard


class SubjVerbExtraction(NamedTuple):
    subject: str
    verb_neg: str
    verb: str

    def __str__(self):
        return ' '.join(self)


def rule(doc, lemmatize=False, verbose=False, letter_case='default') -> List[SubjVerbExtraction]:
    pairs = []
    for verb in doc:
        if verb.pos == VERB:

            if verb.head.dep == nsubj or verb.head.dep == nsubjpass: 
                subject = verb.head

            for child in verb.children:
                if child.dep == nsubj or child.dep == nsubjpass:
                    subject = child

            verb_neg = get_verb_neg(verb)

            if verb_neg is None:
                verb_neg = ''
            else:
                verb_neg = verb_neg.text

            if lemmatize:
                subject = subject.lemma_
                verb = verb.lemma_
            else:
                subject = subject.text
                verb = verb.text

            if letter_case == 'upper':
                subject, verb = subject.upper(), verb.upper()
            elif letter_case == 'lower':
                subject, verb = subject.lower(), verb.lower()

            pairs.append(SubjVerbExtraction(subject=subject, verb_neg=verb_neg, verb=verb))
    return pairs


def extract(input_object, lemmatize: bool = False, want_dataframe: bool = False, verbose: bool = False,
            letter_case: str = 'default'):
    if type(input_object) == str:
        input_object = [input_object, ]
    elif not isinstance(input_object, collections.Iterable):
        raise ValueError('extract_triples: input should be a string or a collection of strings')

    docs = nlp.pipe(input_object)
    pairs = []
    for doc in docs:
        pairs.extend(rule(doc, lemmatize=lemmatize, verbose=verbose, letter_case=letter_case))

    if want_dataframe:
        return pd.DataFrame(pairs)

    return pairs


def extract_df(df, text_column, letter_case: str = 'default', lemmatize: bool = False):
    pair_df_list = []

    def extract_row(row):
        pairs = extract(row[text_column], want_dataframe=True, letter_case=letter_case, lemmatize=lemmatize)
        pairs[list(row.index)] = row
        pair_df_list.append(pairs)

    df.apply(extract_row, axis=1)

    output_df = pd.concat(pair_df_list, axis=0)
    output_df.reset_index(drop=True, inplace=True)
    return output_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='subj_verb_pairs')
    parser.add_argument('input', metavar='input', type=str,
                        help='a filepath to a csv file or an input string')
    parser.add_argument('output', metavar='output', type=str,
                        help='an output path')
    parser.add_argument('--data-column', type=str, default=None, metavar='data_col',
                        help='what column to use if a csv is given', dest='data_column')
    parser.add_argument('--file-delimiter', default='comma', const='comma', nargs='?',
                        choices=['comma', 'pipe', 'tab'],
                        help='delimiter character for data file (default: %(default)s)')
    parser.add_argument('--lemma', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--letter-case', default='default', const='default', nargs='?',
                        choices=['default', 'upper', 'lower'],
                        help='letter casing to use in output (default: %(default)s)')

    args = parser.parse_args()
    is_file = os.path.isfile(args.input)

    inputs = []
    outputs = []

    delimiter = {'comma': ',', 'pipe': '|', 'tab': '\t'}[args.file_delimiter]

    args = parser.parse_args()

    with open(args.output, 'w+') as f:
        pass

    input_values = None
    df = None

    if is_file:
        if args.verbose:
            print('Loading input (%s) as a CSV file...' % args.input)
            print('delimiter:', args.file_delimiter)
        if args.data_column is None:
            exit('Invalid arguments: Must specify column name for data using --data-column')

        usecols = [args.data_column, ]
        df = pd.read_csv(args.input, usecols=usecols, delimiter=delimiter)
        input_values = df[args.data_column]
    else:
        input_values = [args.input, ]

    with open(args.output, 'w+') as f:
        pass

    extraction_count = 0
    header = True

    for i, data_str in enumerate(input_values):
        triples_df = extract(data_str, lemmatize=args.lemma,  verbose=args.verbose, want_dataframe=True,
                             letter_case=args.letter_case)
        extraction_count += len(triples_df)
        if df is not None:
            triples_df['index'] = df.index[i]
        triples_df.to_csv(args.output, mode='a', sep=delimiter, header=header, index=False)

        if header:
            header = False

    if args.verbose:
        print('Number of extractions: %d' % extraction_count)