# Imports
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd
# import mlflow
import os
import pickle
import torch

from frequently_requested_docs.docs_config import MODEL_NAMES

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
    if not os.path.exists("models/" + save_name):
        # download model
        print("Downloading the model, this might take a while...")
        model = SentenceTransformer(model_name)
        
        # save model
        print("Storing model in file")
        model.save("models/" + save_name)
        return model
    else:
        print("Loading model from disc")
        return SentenceTransformer("models/" + save_name)
    
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


# DO NOT USE/CHANGE UNLESS DISCUSSED FIRST, needs to be modified in tandem with loadEmbeddings
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


        # get old embeddings and corpus
        # with open(embedding_path, "rb") as fIn:
        #     cache_data = pickle.load(fIn)
        #     print(type(cache_data['docs']))
        #     new_embeddings = torch.unbind(cache_data['embeddings'])
        #     new_corpus = cache_data['docs']

        #     # add new corpus to embeddings and corpus
        #     add_sentences = add_corpus['Document'].to_list()
        #     new_embeddings += torch.unbind(model.encode(add_sentences, convert_to_tensor=True, show_progress_bar=True))
        #     new_embeddings = torch.stack(new_embeddings)
        #     new_corpus = pd.concat([new_corpus, add_corpus])

        # with open(embedding_path[:-4] + '2' + '.pkl', "wb") as fOut:
        #     pickle.dump({'docs': new_corpus, 'embeddings': new_embeddings}, fOut)

# DOWNLOAD ALL MODELS (DOES NOTHING IF MODELS ALREADY ON DISK)
def downloadAllModels():
    for model_name in MODEL_NAMES:
        model = getModel(model_name, getSaveName(model_name))