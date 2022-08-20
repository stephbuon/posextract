# posextract
posextract offers grammatical information extraction methods designed for the analysis of historical and contemporary textual corpora. It traverses the syntactic dependency relations between parts-of-speech and returns sequences of words that share a grammatical relationship. 

Users have the options of: 

- Extracting subject-verb-object (SVO) and subject-verb-adjective complement (SVA) triples
- Extracting adjective-noun piars
- Extracting subject-verb pairs

## Usage
Required Paramters: 

- `input` can be the name of a csv file or an input string
- `output` name of the output file

Optional Paramters: 
- `--data_column` specify the column to extract triples from
- `--id_column` specify a unique ID field if csv file is given
- `--lemma` specify whether to lemmatize parts-of-speech
- `--post-combine-adj` combine triples (adjective predicate with object)

### Examples

```
python -m posextract.extract_triples "The soldiers were terminally ill." output.csv --post-combine-adj

# Output: soldiers-were-terminally-ill
```

If provided a .csv file: 

```
`python -m posextract.extract_triples --data_column sentence --id_column sentence_id test.csv output.csv`
```

## For More Information...
... see our Wiki: 
- [About Our Evaluation Data](https://github.com/stephbuon/posextract/wiki/Evaluation-Data-Sets)
- [About the Syntactic Dependency Parser](https://github.com/stephbuon/posextract/wiki/Our-Application-of-spaCy-NLP)
