import xgboost as xgb
import pandas as pd
import os
import streamlit as st

class DiseaseModel:

    def __init__(self):
        self.all_symptoms = None
        self.symptoms = None
        self.pred_disease = None
        self.model = xgb.XGBClassifier()
        
        # Define multiple possible base paths for file loading
        self.base_path = os.path.dirname(os.path.dirname(__file__))
        self.possible_paths = [
            self.base_path,
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'Frontend'),
            '/mount/src/multiple-disease-prediction-app/Frontend',
            '.'
        ]
        
        # Load disease list with safe path handling
        self.diseases = self.disease_list(self.safe_path('data/dataset.csv'))

    def safe_path(self, relative_path):
        """Try multiple base paths to find a file"""
        for base_path in self.possible_paths:
            full_path = os.path.join(base_path, relative_path)
            if os.path.exists(full_path):
                return full_path
        
        # If we get here, file wasn't found in any path
        st.warning(f"Could not find file: {relative_path}")
        return relative_path  # Return the original path as fallback

    def load_xgboost(self, model_path):
        # Try multiple possible paths for the model file
        for base_path in self.possible_paths:
            try:
                full_path = os.path.join(base_path, model_path)
                if os.path.exists(os.path.dirname(full_path)):
                    self.model.load_model(full_path)
                    return
            except Exception as e:
                continue
        
        # If we get here, all paths failed
        st.error(f"Could not load model from {model_path}")

    def save_xgboost(self, model_path):
        self.model.save_model(model_path)

    def predict(self, X):
        self.symptoms = X
        disease_pred_idx = self.model.predict(self.symptoms)
        self.pred_disease = self.diseases[disease_pred_idx].values[0]
        disease_probability_array = self.model.predict_proba(self.symptoms)
        disease_probability = disease_probability_array[0, disease_pred_idx[0]]
        return self.pred_disease, disease_probability

    
    def describe_disease(self, disease_name):
        if disease_name not in self.diseases:
            return "That disease is not contemplated in this model"
        
        # Read disease dataframe with safe path handling
        desc_df = pd.read_csv(self.safe_path('data/symptom_Description.csv'))
        desc_df = desc_df.apply(lambda col: col.str.strip())

        return desc_df[desc_df['Disease'] == disease_name]['Description'].values[0]

    def describe_predicted_disease(self):
        if self.pred_disease is None:
            return "No predicted disease yet"

        return self.describe_disease(self.pred_disease)
    
    def disease_precautions(self, disease_name):
        if disease_name not in self.diseases:
            return "That disease is not contemplated in this model"

        # Read precautions dataframe with safe path handling
        prec_df = pd.read_csv(self.safe_path('data/symptom_precaution.csv'))
        prec_df = prec_df.apply(lambda col: col.str.strip())

        return prec_df[prec_df['Disease'] == disease_name].filter(regex='Precaution').values.tolist()[0]

    def predicted_disease_precautions(self):
        if self.pred_disease is None:
            return "No predicted disease yet"

        return self.disease_precautions(self.pred_disease)

    def disease_list(self, kaggle_dataset):
        # Use safe path handling for dataset
        df = pd.read_csv(self.safe_path('data/clean_dataset.tsv'), sep='\t')
        
        # Preprocessing
        y_data = df.iloc[:,-1]
        X_data = df.iloc[:,:-1]

        self.all_symptoms = X_data.columns

        # Convert y to categorical values
        y_data = y_data.astype('category')
        
        return y_data.cat.categories
