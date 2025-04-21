import pandas as pd
import numpy as np
import csv
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier


class HealthCareChatbot:
    """
    Healthcare chatbot for predicting diseases based on symptoms.
    """
    def __init__(self,
                 training_path='Data/Training.csv',
                 desc_path='MasterData/symptom_Description.csv',
                 severity_path='MasterData/symptom_severity.csv',
                 precaution_path='MasterData/symptom_precaution.csv'):
        # Load training data
        self.training = pd.read_csv(training_path)
        self.features = list(self.training.columns[:-1])
        self.X = self.training[self.features]
        self.le = LabelEncoder().fit(self.training['prognosis'])
        self.y = self.le.transform(self.training['prognosis'])

        # Train primary decision tree
        self.model = DecisionTreeClassifier()
        self.model.fit(self.X, self.y)

        # For secondary prediction with string labels
        self.sec_model = DecisionTreeClassifier()
        self.sec_model.fit(self.X, self.training['prognosis'])  # uses string labels

        # Reduced symptom-disease matrix for lookup
        self.reduced_data = self.training.groupby('prognosis').max()

        # Symptom index mapping
        self.symptoms = self.features
        self.symptom_index = {s: i for i, s in enumerate(self.symptoms)}

        # Load master dictionaries
        self.description = self._load_csv_dict(desc_path, key_col=0, val_col=1)
        self.severity = self._load_csv_dict(severity_path, key_col=0, val_col=1, val_type=int)
        self.precautions = self._load_csv_precautions(precaution_path)

    def _load_csv_dict(self, path, key_col=0, val_col=1, val_type=str):
        d = {}
        with open(path, newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                try:
                    d[row[key_col]] = val_type(row[val_col])
                except Exception:
                    continue
        return d

    def _load_csv_precautions(self, path):
        d = {}
        with open(path, newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                # Expecting: disease,prec1,prec2,prec3,prec4
                d[row[0]] = row[1:]
        return d

    def calculate_condition(self, symptoms, days):
        total_severity = sum(self.severity.get(s, 0) for s in symptoms)
        score = (total_severity * days) / (len(symptoms) + 1)
        if score > 13:
            return 'High risk: consider consulting a doctor.'
        else:
            return 'Low risk: monitor symptoms and take precautions.'

    def sec_predict(self, symptoms):
        # Secondary prediction using string-labeled model
        vec = np.zeros(len(self.symptoms))
        for s in symptoms:
            idx = self.symptom_index.get(s)
            if idx is not None:
                vec[idx] = 1
        return self.sec_model.predict([vec])[0]

    def get_symptom_list_for_disease(self, disease):
        # Symptoms commonly associated with a disease
        row = self.reduced_data.loc[disease]
        present = row[row > 0].index.tolist()
        return present

    def get_chatbot_response(self, initial_symptom, days, additional_symptoms=None):
        """
        Given an initial symptom and days duration (plus optional follow-up symptoms),
        returns a dictionary with prediction, descriptions, precautions, and advice.
        """
        # Primary pathway: gather related symptoms
        related = self.get_symptom_list_for_disease(self.le.inverse_transform([self.model.predict(self.X.iloc[0:1])[0]])[0])

        # Combine symptoms
        symptoms = [initial_symptom] + (additional_symptoms or [])

        # Secondary disease prediction
        disease = self.sec_predict(symptoms)

        # Severity assessment
        risk_advice = self.calculate_condition(symptoms, days)

        # Descriptions and precautions
        descr = self.description.get(disease, 'No description available.')
        precs = self.precautions.get(disease, [])

        return {
            'disease': disease,
            'description': descr,
            'precautions': precs,
            'risk_advice': risk_advice,
            'symptoms_checked': symptoms,
            'related_symptoms': related
        }


# Caller function for Streamlit
# Example usage in Streamlit app:
# >>> from healthcare_chatbot_module import get_chatbot_response, chatbot
# >>> response = chatbot.get_chatbot_response('fever', days=3, additional_symptoms=['fatigue'])

# Singleton instance
chatbot = HealthCareChatbot()

def get_chatbot_response(initial_symptom, days, additional_symptoms=None):
    """
    Streamlit caller: initial_symptom (str), days (int), additional_symptoms (list of str)
    Returns response dict with:
      - disease
      - description
      - precautions
      - risk_advice
      - symptoms_checked
      - related_symptoms
    """
    return chatbot.get_chatbot_response(initial_symptom, days, additional_symptoms)
