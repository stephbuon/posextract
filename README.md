# posextract
posextract offers grammatical information extraction methods designed for the analysis of historical and contemporary textual corpora. It traverses the syntactic dependency relations between parts-of-speech and returns sequences of words that share a grammatical relationship. See [our article]() for more. You can also [download posextract for pypi with pip](https://pypi.org/project/posextract/). 

## Usage

- `extract_triples` to extract subject-verb-object (SVO) and subject-verb-adjective complement (SVA) triples
- `extract_adj_noun_pairs` to extract adjective-noun pairs
- `extract_subj_verb_pairs` to extract subject-verb pairs

Required Paramters: 

- `input` can be the name of a csv file or an input string
- `output` name of the output file

Optional Paramters: 
- `--data_column` specify the column to extract triples from
- `--id_column` specify a unique ID field if csv file is given
- `--lemma` specify whether to lemmatize parts-of-speech
- `--post-combine-adj` combine triples (adjective predicate with object)

### Examples

#### Interactive: 

Extract grammatical triples.

```
from posextract import extract_triples

triples = grammatical_triples.extract(['Landlords may exercise oppression.', 'The soldiers were ill.'])

for triple in triples:
    print(triple)

# Output: Landlords exercise oppression, soldiers were ill
```

Or extract adjectives and the nouns they modify. 

```
from posextract import adj_noun_pairs

adj_noun = adj_noun_pairs.extract()
```

Or extract subjects and their verbs. 

```
from posextract import subj_verb_pairs

subj_verb = subj_verb_pairs.extract()
```

#### Over CLI: 

posextract can extract grammatical triples from text: 

```
python -m posextract.extract_triples "Landlords may exercise oppression." output.csv

# Output: Landlords exercise oppression
```

posextract can extract SVO/SVA relationships separately or it can combine the adjective as part of a SVO triple:

```
python -m posextract.extract_triples "The soldiers were terminally ill." output.csv --post-combine-adj

# Output: soldiers were terminally, soldiers were ill 
```

```
python -m posextract.extract_triples "The soldiers were terminally ill." output.csv --post-combine-adj

# Output: soldiers were terminally ill
```

If provided a .csv file: 

```
python -m posextract.extract_triples --data_column sentence --id_column sentence_id input.csv output.csv
```

## For More Information...
... see our Wiki: 
- [About Our Evaluation Data](https://github.com/stephbuon/posextract/wiki/Evaluation-Data-Sets)
- [About the Syntactic Dependency Parser](https://github.com/stephbuon/posextract/wiki/Our-Application-of-spaCy-NLP)
