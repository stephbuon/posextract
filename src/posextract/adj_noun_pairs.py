import argparse
import warnings; warnings.simplefilter('ignore')
import pandas as pd
import spacy
from spacy.symbols import amod, acomp, NOUN

def rule(doc):
  pairs = []
  for adjective in doc:
    if adjective.dep == amod or adjective.dep == acomp and adjective.head.pos == NOUN: # or adjective.dep == ccomp or adjective.dep == conj
      #concat_extracted_pairs = ' '.join(adjective.text, adjective.head.lemma_)
      pairs.append(str(' '.join([adjective.text, adjective.head.lemma_])))
  return pairs

def extract(hansard, col, **kwargs):

  nlp = spacy.load('en_core_web_sm', disable = ['tagger', 'ner', 'attribute_ruler'])
    
  kw = kwargs.get('keep', None)
    
  if kw == 'keep':
    hansard['parsed_text'] = [doc for doc in nlp.pipe(hansard[col].tolist())] # this turns into env
    hansard['adj_noun_pair'] = hansard['parsed_text'].apply(rule)
    hansard = hansard.loc[:, hansard.columns != 'parsed_text']
  else:
    hansard[col] = [doc for doc in nlp.pipe(hansard[col].tolist())] 
    hansard['adj_noun_pair'] = hansard[col].apply(rule)
    hansard = hansard.loc[:, hansard.columns == 'adj_noun_pair']

  hansard = hansard[hansard.astype(str)['adj_noun_pair'] != '[]']

  hansard = hansard.explode('adj_noun_pair')

  return hansard

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  
  parser.add_argument('dataset',
                      help='Name of file to extract triples from.')
  parser.add_argument('col',
                      help='Name of column to extract triples from.')

  args = parser.parse_args()

  extract(args.dataset, args.col)
