### File Contents


**generate_wikipedia_dataset.py** 

Create a new data set of random sentences from Wikipedia.

| Argument  | Definition |
| ------------- | ------------- |
| --input_number | Number of sentences to generate. |
| --output  | Output file path (include name of output file in dir path) |

Example Useage:
`python3 generate_wikipedia_dataset.py 50000 /home/stephbuon/projects/posextract/evaluation/wikipedia_dataset.csv`

**select_sentences.py** 

Take an existing data set and export sentences that meet the criteria for this study (e.g. random sentences, sentences with leftward syntactic movement, more)

| Argument  | Definition |
| ------------- | ------------- |
| --input_number | Number of sentences to generate. |
| --output  | Output file path (include name of output file in dir path) |

Example Useage:
`python3 select_sentences.py`