import spacy
import warnings; warnings.simplefilter('ignore')
import pandas as pd
from spacy.symbols import amod, acomp, NOUN

def extract_adj_noun_pairs(doc):
    pairs = []
    for adjective in doc:
      if adjective.dep == amod or adjective.dep == acomp and adjective.head.pos == NOUN: # or adjective.dep == ccomp or adjective.dep == conj
        #concat_extracted_pairs = ' '.join(adjective.text, adjective.head.lemma_)
        pairs.append(str(' '.join([adjective.text, adjective.head.lemma_])))
    return pairs

def driver(hansard, col, **kwargs):

    nlp = spacy.load('en_core_web_sm', disable = ['tagger', 'ner', 'attribute_ruler'])
    
    kw = kwargs.get('keep', None)
    
    if kw == 'keep':
      hansard['parsed_text'] = [doc for doc in nlp.pipe(hansard[col].tolist())] # this turns into env
      hansard['adj_noun_pair'] = hansard['parsed_text'].apply(extract_adj_noun_pairs)
      hansard = hansard.loc[:, hansard.columns != 'parsed_text']
    else:
      hansard[col] = [doc for doc in nlp.pipe(hansard[col].tolist())] 
      hansard['adj_noun_pair'] = hansard[col].apply(extract_adj_noun_pairs)
      hansard = hansard.loc[:, hansard.columns == 'adj_noun_pair']

    hansard = hansard[hansard.astype(str)['adj_noun_pair'] != '[]']

    hansard = hansard.explode('adj_noun_pair')

    return(hansard)
