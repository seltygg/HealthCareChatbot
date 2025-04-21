import pandas as pd
import numpy as np
import csv
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.agents import Tool, initialize_agent, AgentType

class HealthCareChatbot:
    """
    Modular healthcare chatbot combining Decision Trees for disease prediction
    with Retrieval-Augmented Generation (RAG) and AI agents for detailed
    medical information.
    """
    def __init__(self,
                 training_path='Data/Training.csv',
                 desc_path='MasterData/symptom_Description.csv',
                 severity_path='MasterData/symptom_severity.csv',
                 precaution_path='MasterData/symptom_precaution.csv'):
        # --- Decision Tree Setup ---
        self.training = pd.read_csv(training_path)
        self.features = list(self.training.columns[:-1])
        self.X = self.training[self.features]
        self.le = LabelEncoder().fit(self.training['prognosis'])
        self.y = self.le.transform(self.training['prognosis'])
        self.model = DecisionTreeClassifier().fit(self.X, self.y)
        self.sec_model = DecisionTreeClassifier().fit(
            self.X, self.training['prognosis']
        )
        self.reduced_data = self.training.groupby('prognosis').max()
        self.symptom_index = {s: i for i, s in enumerate(self.features)}

        # --- Master Data Loaders ---
        self.description = self._load_csv_dict(desc_path, 0, 1)
        self.severity = self._load_csv_dict(severity_path, 0, 1, int)
        self.precautions = self._load_csv_precautions(precaution_path)

        # --- RAG Infrastructure ---
        loader = CSVLoader(file_path=desc_path, csv_args={'delimiter': ','})
        docs = loader.load()
        embeddings = OpenAIEmbeddings()
        self.vectorstore = FAISS.from_documents(docs, embeddings)
        self.llm = ChatOpenAI(temperature=0)
        self.rag_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(),
            return_source_documents=True
        )

        # --- Tools & Agent ---
        predict_tool = Tool(
            name="disease_predictor",
            func=self._predict_tool,
            description="Predict disease from symptoms"
        )
        rag_tool = Tool(
            name="medical_lookup",
            func=self._rag_tool,
            description="Fetch detailed medical info via RAG"
        )
        self.agent = initialize_agent(
            [predict_tool, rag_tool],
            self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False
        )

    def _load_csv_dict(self, path, key_col, val_col, val_type=str):
        d = {}
        with open(path, newline='') as f:
            for row in csv.reader(f):
                try:
                    d[row[key_col]] = val_type(row[val_col])
                except:
                    continue
        return d

    def _load_csv_precautions(self, path):
        d = {}
        with open(path, newline='') as f:
            for row in csv.reader(f):
                d[row[0]] = row[1:]
        return d

    def _predict_tool(self, query: str) -> str:
        """
        Expects a comma-separated list of symptoms with optional days.
        E.g. "fever, fatigue; days=3"""
        parts = query.split(';')
        syms = [s.strip() for s in parts[0].split(',')]
        days = int(parts[1].split('=')[1]) if len(parts) > 1 else 1
        vec = np.zeros(len(self.features))
        for s in syms:
            idx = self.symptom_index.get(s)
            if idx is not None:
                vec[idx] = 1
        primary = self.model.predict([vec])[0]
        secondary = self.sec_model.predict([vec])[0]
        advice = self._calc_risk(syms, days)
        return f"Primary: {self.le.inverse_transform([primary])[0]}, " + \
               f"Secondary: {secondary}, Advice: {advice}"

    def _rag_tool(self, query: str) -> str:
        """Use RAG chain to fetch detailed medical info."""
        result = self.rag_chain({"question": query, "chat_history": []})
        return result['answer']

    def _calc_risk(self, symptoms, days):
        total = sum(self.severity.get(s, 0) for s in symptoms)
        score = (total * days) / (len(symptoms) + 1)
        return 'High risk' if score > 13 else 'Low risk'

    def ask(self, prompt: str) -> str:
        """
        Top-level method for Streamlit: passes user prompt to AI agent.
        """
        return self.agent.run(prompt)

# Singleton and caller
chatbot = HealthCareChatbot()

def get_response(prompt: str) -> str:
    """Streamlit caller: sends full user prompt to AI agent."""
    return chatbot.ask(prompt)
