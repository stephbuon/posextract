`generate_wikipedia_data_set.py` : use to create wikipedia data set

arguments:
  --input_number
  --output

export_sentences.py : take a data set and look for sentences meeting criteria



Examples: 

collect_sentences('/scratch/group/pract-txt-mine/hansard_justnine_12192019.csv', 'text')




Detailed Usage
---------------
**Usage: generate_wikipedia_dataset.py**

optional arguments:
  -h, --help            show this help message and exit
  --corpus-dir CORPUS_DIR
                        Corpus directory
  --file-extension EXT  Corpus file extension
  --working-dir WORKING_DIR
                        Working directory
  --output-dir OUTPUT_DIR
                        Output directory
  --context-size WINDOW
                        Context size to use for training embeddings
  --epochs EPOCHS       Number of epochs to training embeddings
  --start-time-point START
                        Start time point
  --end-time-point END  End time point
  --step-size STEP      Step size for timepoints
  --model-family MODEL_FAMILY
                        Model family default (locallinear)
  --number-nearest-neighbors KNN 
                        Number of nearest neighbors to use for mapping to
                        joint space (default:1000)
                          --vocabulary-file VOCAB_FILE
                        Common vocabulary file
  --threshold THRESHOLD
                        Threshold for mean shift model for change point
                        detection (default: 1.75)
  --bootstrap-samples BOOTSTRAP
                        Number of bootstrap samples to draw (default: 1000)
  --workers WORKERS     Maximum number of workers (default: 1)
  -l LOG, --log LOG     log verbosity level
 
