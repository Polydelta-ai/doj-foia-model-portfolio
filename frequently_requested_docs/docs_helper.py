# Imports
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd
# import mlflow
import os
import pickle
import torch

from frequently_requested_docs.docs_config import MODEL_NAMES

def test_sentence(sentence, model, corpus_docs, corpus_embeddings, TOP_K):

    # encode sentence to get sentence embeddings
    sentence_embedding = model.encode(sentence, convert_to_tensor=True)

    # compute similarity scores of the sentence with the corpus
    cos_scores = util.pytorch_cos_sim(sentence_embedding, corpus_embeddings)[0]

    # Sort the results in decreasing order and get the first TOP_K
    top_results = np.argpartition(-cos_scores, range(TOP_K))[0:TOP_K]

    print("Sentence:", sentence, "\n")
    print("Top", TOP_K, "most similar sentences in corpus:")
    for idx in top_results[0:TOP_K]:
        print(corpus_docs.iloc[int(idx)]["Document"], "(Score: %.4f)" % (cos_scores[idx]))


def getSaveName(model_name):
    ret = ""
    for i, letter in enumerate(model_name):
        if letter != "/":
            ret += letter
        else:
            ret += "-"
    return ret

def getEmbeddingPath(save_name):
    return "frequently_requested_docs/embeddings/" + save_name + ".pkl"

def getModel(model_name, save_name):
    # if model not saved 
    if not os.path.exists("frequently_requested_docs/models/" + save_name):
        # download model
        print("Downloading the model, this might take a while...")
        model = SentenceTransformer(model_name)
        
        # save model
        print("Storing model in file")
        model.save("frequently_requested_docs/models/" + save_name)
        return model
    else:
        print("Loading model from disc")
        return SentenceTransformer("frequently_requested_docs/models/" + save_name)
    
def loadEmbeddings(model, embedding_path, corpus_docs):
    # if path doesnt exist, 
    if not os.path.exists(embedding_path):
        # read your corpus etc
        corpus_sentences = [row["Document"] for row in corpus_docs]
        print("Encoding the corpus. This might take a while")
        # encode corpus to get corpus embeddings
        corpus_embeddings = model.encode(corpus_sentences, convert_to_tensor=True, show_progress_bar=True)

        print("Storing embeddings in file")
        with open(embedding_path, "wb") as fOut:
            pickle.dump({'docs': corpus_docs, 'embeddings': corpus_embeddings}, fOut)
        
        return (corpus_docs, corpus_embeddings)

    else:
        print("Loading pre-computed embeddings from disc")
        with open(embedding_path, "rb") as fIn:
            cache_data = pickle.load(fIn)
            corpus_docs = cache_data['docs']
            corpus_embeddings = cache_data['embeddings']
        
        return (corpus_docs, corpus_embeddings)


# define a new structure for saving embeddings
# currently saved as (corpus_docs, embeddings) pairs
def changeEmbeddingSave(corpus_docs):
    for model_name in MODEL_NAMES:
        new_path = "embeddings/" + getSaveName(model_name) + ".pkl"
        embeddings = []
        with open(new_path, "rb") as fIn:
            cache_data = pickle.load(fIn)
            embeddings = cache_data['embeddings']
        with open(new_path, "wb") as fOut:
            pickle.dump({'docs': corpus_docs, 'embeddings': embeddings}, fOut)

# slow, re-encodes entire corpus
def addAgencytoSavedEmbeddings():
    corpus_docs = pd.read_csv('frequently_requested_docs.csv')
    for model_name in MODEL_NAMES:
        save_name = getSaveName(model_name)
        embedding_path = "embeddings/" + save_name + ".pkl"

        model = getModel(model_name, save_name)

        corpus_sentences = corpus_docs['Document'].to_list()
        # encode corpus to get corpus embeddings
        corpus_embeddings = model.encode(corpus_sentences, convert_to_tensor=True, show_progress_bar=True)

        print("Storing embeddings in file")
        with open(embedding_path, "wb") as fOut:
            pickle.dump({'docs': corpus_docs, 'embeddings': corpus_embeddings}, fOut)


# DOWNLOAD ALL MODELS (DOES NOTHING IF MODELS ALREADY ON DISK)
def downloadAllModels():
    for model_name in MODEL_NAMES:
        model = getModel(model_name, getSaveName(model_name))
