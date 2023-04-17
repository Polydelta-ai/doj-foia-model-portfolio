# DEFINE CONSTANTS

# number of documnt 
TOP_K = 25

# names of tested models (pulled from SentenceTransformers)
MODEL_NAMES = [
    'nli-mpnet-base-v2',
    'nli-roberta-base-v2',
    'princeton-nlp/sup-simcse-roberta-large',
    'princeton-nlp/unsup-simcse-roberta-large',
    'stsb-distilroberta-base-v2',
    'stsb-mpnet-base-v2',
    'stsb-roberta-base',
    'stsb-roberta-base-v2',
    'stsb-roberta-large',
]

DATA_CSV_PATH = 'frequently_requested_docs/frequently_requested_docs.csv'

TEST_DATA_CSV_PATH = 'test_sentences.csv'

DATA_ROW_NAMES = {
    'component': "Component",
    'document': 'Document',
    'tag': 'Tag',
    'url': 'URL',
    'agency': 'Agency',
}

# similarity threshold for counting results as true/false pos/neg
SIM_THRESH = 0.7

