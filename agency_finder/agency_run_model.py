"""
Module to run sepCNN model
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time

import tensorflow as tf
import numpy as np

# import build_model
import agency_finder.agency_load_data
import agency_finder.agency_vectorize_data
import agency_finder.agency_explore_data

from agency_finder.agency_config import COMPONENT_DICT

FLAGS = None

# return formatted predictions, giving actual agency based on component_dict
def inference(model, inputs):
        
    inputs = [inputs]
    predictions = model.predict(np.array(inputs))

    formatted = []
    for i in range(len(inputs)):
        prediction = predictions[i]
        sorted = []
        for ind, val in enumerate(prediction):
            sorted.append((ind, val))
        
        sorted.sort(key=lambda x: x[1], reverse=True)
        sorted = [(COMPONENT_DICT[component[0]], component[1]) for component in sorted]
        formatted.append((inputs[i], sorted))

    return formatted

def getModel(model_path):
    model = tf.keras.models.load_model(f'agency_finder/models/agency_finder_sepcnn_model_{model_path}.tf')
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='agency_finder_sepcnn_model',
                        help='input model file name (no ext)')
    FLAGS, unparsed = parser.parse_known_args()

    model_path = f'../models/{FLAGS.model}.tf'

    model = getModel(model_path)

    examples = ['I am searching for the Detention Facility Reviews for the Randall County Jail in Amarillo, Texas', 'Statements made by former georgia senator david perdue about visas.', 'All documents regarding the TSA’s throughput data for August 2017', 'A list of everyone that applied to Lead positions at LaGuardia Airport, NY. (Date Range for Record  Search: From 01/01/2017 To 03/01/2020)']

    inputs = [examples[3]]

    # print(model.predict(np.array(inputs)))
    predictions = inference(model, inputs)
    print(predictions)

