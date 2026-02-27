# ============================================
# IMPORTS
# ============================================
import pandas as pd
import numpy as np
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, classification_report
nltk.download('stopwords')

# Ensure stopwords are available, download only if missing
def ensure_stopwords():
    import nltk
    try:
        from nltk.corpus import stopwords
        _ = stopwords.words('english')
    except LookupError:
        print('Downloading NLTK stopwords...')
        nltk.download('stopwords')

ensure_stopwords()


# ============================================
# TEXT PREPROCESSOR CLASS
# ============================================
class TextPreprocessor:

    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.ps = PorterStemmer()

    def clean_html(self, text):
        clean = re.compile('<.*?>')
        return re.sub(clean, '', text)

    def remove_special(self, text):
        return ''.join(char for char in text if char.isalnum() or char == ' ')

    def preprocess(self, text):

        if not isinstance(text, str):
            text = str(text)

        # cleaning steps
        text = self.clean_html(text)
        text = text.lower()
        text = self.remove_special(text)

        words = text.split()

        # stopword removal
        words = [w for w in words if w not in self.stop_words]

        # stemming
        words = [self.ps.stem(w) for w in words]

        return " ".join(words)


# ============================================
# DATA LOADER CLASS
# ============================================
class DatasetLoader:

    def __init__(self, path):
        self.path = path

    def load_data(self):
        df = pd.read_csv(self.path)


        # sample for faster training
        df = df.sample(2000, random_state=42).reset_index(drop=True)

        # label encoding
        df['sentiment'] = df['sentiment'].replace({
            'positive': 1,
            'negative': 0
        })

        # Ensure sentiment is integer type
        df['sentiment'] = df['sentiment'].astype(int)

        return df


# ============================================
# MODEL CLASS
# ============================================
class SentimentModel:

    def __init__(self):
        self.vectorizer = CountVectorizer()
        self.models = {
            "MultinomialNB": MultinomialNB(),
            "BernoulliNB": BernoulliNB()
        }

    def vectorize(self, reviews):
        return self.vectorizer.fit_transform(reviews).toarray()

    def train(self, X_train, y_train):
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            print(f"{name} trained successfully")

    def evaluate(self, X_test, y_test):

        for name, model in self.models.items():
            preds = model.predict(X_test)

            print("\n======================")
            print(f"Model: {name}")
            print("Accuracy:", accuracy_score(y_test, preds))
            print(classification_report(y_test, preds))


# ============================================
# MAIN PIPELINE (OOP FLOW)
# ============================================
class SentimentPipeline:

    def __init__(self, dataset_path):
        self.loader = DatasetLoader(dataset_path)
        self.preprocessor = TextPreprocessor()
        self.model = SentimentModel()

    def run(self):

        print("Loading Dataset...")
        df = self.loader.load_data()

        print("Preprocessing Text...")
        df['review'] = df['review'].apply(self.preprocessor.preprocess)

        print("Vectorizing Text...")
        X = self.model.vectorize(df['review'])
        y = df['sentiment']

        print("Train-Test Split...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print("Training Models...")
        self.model.train(X_train, y_train)

        print("Evaluating Models...")
        self.model.evaluate(X_test, y_test)


# ============================================
# RUN PROGRAM
# ============================================
if __name__ == "__main__":

    pipeline = SentimentPipeline("Vivek Kumar\\Supervised Learning\\NaiveBayes\\IMDB Dataset.csv")
    pipeline.run()