import os
import pickle
import shutil
import numpy as np

from pathlib import Path

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from structures import FitConfig, PredictConfig, Texts, Labels, Scores, Prediction
from os import environ as env

MODEL_DIR = env['MODEL_DIR']
MODEL_DIR_LOADED = env['MODEL_DIR_LOADED']

class TextClassifier:
    @staticmethod
    def fit(texts: Texts, labels: Labels, config: FitConfig) -> None:

        if config.feature_type == 'tf-idf':
            vectorizer = TfidfVectorizer()

        elif config.feature_type == 'bow':
            vectorizer = CountVectorizer()

        else:
            raise ValueError(f'Unknown config.feature_type: "{config.feature_type}"')

        data = vectorizer.fit_transform(texts.values)

        if config.model_clf == 'logistic-regression':
            model = LogisticRegression()

        elif config.model_clf == 'boosting':
            model = GradientBoostingClassifier(n_estimators=300000)

        elif config.model_clf == 'random-forest':
            model = RandomForestClassifier(n_estimators=200000)

        else:
            raise ValueError(f'Unknown config.model_clf: "{config.model_clf}"')

        model.fit(data, labels.values)

        root_path = Path(MODEL_DIR)
        if not Path(MODEL_DIR).exists():
            root_path.mkdir()

        model_path = Path(os.path.join(MODEL_DIR, config.model_name))

        if model_path.exists():
            shutil.rmtree(model_path)

        model_path.mkdir()

        with open(model_path / 'model.pkl', 'wb') as fout:
            pickle.dump(model, fout)

        with open(model_path / 'vectorizer.pkl', 'wb') as fout:
            pickle.dump(vectorizer, fout)

    @staticmethod
    def predict(texts: Texts, config: PredictConfig):

        model_path = Path(os.path.join(MODEL_DIR_LOADED, config.model_name))

        if not model_path.exists() or not model_path.is_dir():
            raise ValueError(
                f'Path "{model_path}" is not a valid path to model')

        if not (model_path / 'model.pkl').exists() or not (
                model_path / 'vectorizer.pkl').exists():
            raise ValueError(f'Model from "{model_path}" is corrupted')

        if config.top_n <= 0:
            raise ValueError(
                f'Top n value "{config.top_n}" must be positive int')

        with open(model_path / 'model.pkl', 'rb') as fin:
            model = pickle.load(fin)

        with open(model_path / 'vectorizer.pkl', 'rb') as fin:
            vectorizer = pickle.load(fin)

        scores_list = model.predict_proba(vectorizer.transform(texts.values))

        labels_list_ = []
        scores_list_ = []

        for scores in scores_list:
            sorted_scores = list(np.sort(scores))[::-1]
            sorted_labels = [model.classes_[i] for i in np.argsort(scores)][
                            ::-1]
            labels_list_.append(sorted_labels[: config.top_n])
            scores_list_.append(sorted_scores[: config.top_n])
        return Prediction(
            labels_list=[Labels(values=labels) for labels in labels_list_],
            scores_list=[Scores(values=scores) for scores in scores_list_],
        )

