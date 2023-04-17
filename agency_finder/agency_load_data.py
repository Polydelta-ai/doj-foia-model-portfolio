"""Module to load data.
Consists of functions to load data from four different datasets (IMDb, Rotten
Tomatoes, Tweet Weather, Amazon Reviews). Each of these functions do the
following:
    - Read the required fields (texts and labels).
    - Do any pre-processing if required. For example, make sure all label
        values are in range [0, num_classes-1].
    - Split the data into training and validation sets.
    - Shuffle the training data.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random

import numpy as np
import pandas as pd

from agency_finder.agency_config import COMPONENT_DICT

def load_agency_finder_dataset(data_path,
                               file_name,
                               validation_split=0.2,
                               seed=123):
    """Loads the rotten tomatoes sentiment analysis dataset.
    # Arguments
        data_path: string, path to the data directory.
        validation_split: float, percentage of data to use for validation.
        seed: int, seed for randomizer.
    """
    header = ['request', 'component']

    if not os.path.exists(os.path.join(data_path, f'{file_name[:-4]}_shuffled.csv')):
        columns = (1, 3)  # 1 - Request, 3 - Agency Component
        data = _load_and_shuffle_data(data_path, file_name, columns, seed, ',')


        # Get the request and agency values
        texts = np.array(data['request'])
        labels = np.array(data['component'])

        (train_texts, train_labels), (test_texts, test_labels) = split_training_and_validation_sets(texts, labels, validation_split)

        # Save train and test splits for consistent model comparisons
        shuffled_data = [[texts[i], labels[i]] for i in range(len(texts))]
        shuffled_data = pd.DataFrame(shuffled_data, columns=header)
        shuffled_data.to_csv(f'../data/{file_name[:-4]}_shuffled.csv', index=False)

        return (train_texts, train_labels), (test_texts, test_labels)
    else:
        shuffled_data = pd.read_csv(f'../data/{file_name[:-4]}_shuffled.csv', header=0)
        
        texts = np.array(shuffled_data['request'])
        labels = np.array(shuffled_data['component'])

        return split_training_and_validation_sets(texts, labels, validation_split)
        


def _load_and_shuffle_data(data_path,
                           file_name,
                           cols,
                           seed,
                           separator=',',
                           header=0):
    """Loads and shuffles the dataset using pandas.
    # Arguments
        data_path: string, path to the data directory.
        file_name: string, name of the data file.
        cols: list, columns to load from the data file.
        seed: int, seed for randomizer.
        separator: string, separator to use for splitting data.
        header: int, row to use as data header.
    """
    np.random.seed(seed)
    data_path = os.path.join(data_path, file_name)
    data = pd.read_csv(data_path, usecols=cols, sep=separator, header=header)
    return data.reindex(np.random.permutation(data.index))


def split_training_and_validation_sets(texts, labels, validation_split):
    """Splits the texts and labels into training and validation sets.
    # Arguments
        texts: list, text data.
        labels: list, label data.
        validation_split: float, percentage of data to use for validation.
    # Returns
        A tuple of training and validation data.
    """
    component_texts = [[] for i in COMPONENT_DICT]
    # place each text in a bucket for its component
    for i in range(len(texts)):
        component_texts[labels[i]].append(texts[i])

    train_texts = []
    train_labels = []
    test_texts = []
    test_labels = []

    # add texts to train and test sets based validation_split
    for i in range(len(component_texts)):
        num_training_samples = int((1 - validation_split) * len(component_texts[i]))
        train_texts += component_texts[i][:num_training_samples]
        train_labels += [i for x in component_texts[i][:num_training_samples]]
        test_texts += component_texts[i][num_training_samples:]
        test_labels += [i for x in component_texts[i][num_training_samples:]]

    return ((train_texts, train_labels), (test_texts, test_labels))