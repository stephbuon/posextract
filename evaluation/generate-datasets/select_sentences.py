#add nchar greater than 40

import spacy
import os
import re
import pandas as pd

nlp = spacy.load('en_core_web_sm')

def validate_data(df, col):
    df = df[df[col].str.match(r'hon\.$') == False]
    df = df[df[col].str.match(r'^(?![A-Z])') == False]
    df['num_words'] = df[col].str.split().str.len()
    #df = df[df['num_words'] > 4]    
    return df

def syntax_check(doc):
    sent_pos = ' '.join([token.pos_ for token in doc])
    if 'VERB' in sent_pos and 'NOUN' in sent_pos or 'PRON':
        if sent_pos.count('NOUN') >= 2:
            return 'valid'
    else:
        return 'invalid'

def validate_syntax(df):
    df['syntax_check'] = df['parsed_text'].apply(syntax_check)
    df = df[df['syntax_check'] == 'valid']
    del df['syntax_check']
    return df

def sentence_check(doc):
    if re.match(r'^Which (.*)\?$|^What (.*)\?$|^Why (.*)\?$|^Where (.*)\?$|^When (.*)\?$', str(doc), re.IGNORECASE):
        if doc[1].pos_ == 'NOUN':
            return 'interrogative_sent'
    elif str(doc).count(',') > 0:
        return 'comp_sent'
    elif doc[0].pos_ != 'NOUN' and doc[0].pos_ != 'PRON' and doc[0].pos_ != 'PROPN':
        if str(doc[0]) != 'The':
            if doc[1].pos_ == 'VERB' or doc[1].pos_ == 'ADJ':
                return 'leftward_sent'

def tag_sentence(df):  
    df['tag'] = df['parsed_text'].apply(sentence_check)
    del df['parsed_text']
    return df

def collect_sentences(df, col):
    #df = pd.read_csv(df)
    df = pd.read_csv(df, sep='|')
    df = df[[col]].copy()
        
    df = validate_data(df, col)
    
    df['parsed_text'] = list(nlp.pipe(df[col], disable = ['ent']))
    
    df = validate_syntax(df)
    df = tag_sentence(df)

    export_dir = '/home/stephbuon/projects/posextract/evaluation'
    #export_dir = '/users/sbuongiorno/sentence_eval'
    #if not os.path.exists(export_dir):
    #    os.makedirs(export_dir)
        
    types = ['leftward_sent', 'interrogative_sent', 'comp_sent']
                
    for t in types:
        out = df[df['tag'] == t]
        try:
            out = out.sample(10)
            out.to_csv(export_dir + '/' + t + '.csv')
        except:
            continue
            
    df = df.sample(n = 1500)
    df.to_csv(export_dir + '/' + 'random_sent.csv')
    
#collect_sentences('/scratch/group/pract-txt-mine/hansard_justnine_12192019.csv', 'text')

collect_sentences('/home/stephbuon/projects/posextract/evaluation/wikipedia_sentences.csv', 'text')