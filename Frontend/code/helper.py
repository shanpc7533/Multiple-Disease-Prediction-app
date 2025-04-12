import pandas as pd
import numpy as np
import os
import streamlit as st


def prepare_symptoms_array(symptoms):
    '''
    Convert a list of symptoms to a ndim(X) (in this case 133) that matches the
    dataframe used to train the machine learning model

    Output:
    - X (np.array) = X values ready as input to ML model to get prediction
    '''
    symptoms_array = np.zeros((1,133))
    
    # Define multiple possible base paths for file loading
    base_path = os.path.dirname(os.path.dirname(__file__))
    possible_paths = [
        base_path,
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'Frontend'),
        '/mount/src/multiple-disease-prediction-app/Frontend',
        '.'
    ]
    
    # Try to find the dataset in multiple possible locations
    dataset_path = None
    for path in possible_paths:
        try:
            test_path = os.path.join(path, 'data/clean_dataset.tsv')
            if os.path.exists(test_path):
                dataset_path = test_path
                break
        except Exception:
            continue
    
    if dataset_path is None:
        # Silent error handling - don't show errors on the app
        return symptoms_array
    
    # Load the dataset
    try:
        df = pd.read_csv(dataset_path, sep='\t')
        
        for symptom in symptoms:
            symptom_idx = df.columns.get_loc(symptom)
            symptoms_array[0, symptom_idx] = 1
            
    except Exception as e:
        # Silent error handling - don't show errors on the app
        pass
    
    return symptoms_array
