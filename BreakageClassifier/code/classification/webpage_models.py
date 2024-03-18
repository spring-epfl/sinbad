from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.calibration import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler
from imblearn.over_sampling import SMOTE


class Preprocessor:
    def __init__(self, features):
        self.threshold_features = []
        self.constant_features = []
        self.original_features = features

    def _features_to_threshold(self, df):
        _thresholds = []
        _new_features = []

        for feature in self.features:
            if df[feature].dtype == np.float64 and df[feature].nunique() > 10:
                _thresholds.append(feature)
                _new_features.append(f"{feature}_thr0")
                _new_features.append(f"{feature}_thr1")
                _new_features.append(f"{feature}_thr0.5")

        return _thresholds, _new_features

    def _constant_features(self, df):
        return [
            col
            for col in df.columns
            if len(df[col].unique()) == 1 and col not in ["issue", "is_breaking"]
        ]

    def threshold(self, df):
        for feature in self.threshold_features:
            df[f"{feature}_thr0"] = df[feature] > 0
            df[f"{feature}_thr1"] = df[feature] > 1
            df[f"{feature}_thr0.5"] = df[feature] > 0.5

        return df

    def remove_constant(self, df):
        return df.drop(columns=self.constant_features)

    def fit_transform(self, df):
        self.features = list(set(self.original_features) & set(df.columns))

        # self.threshold_features, new_features = self._features_to_threshold(df)
        # self.features += new_features

        self.constant_features = self._constant_features(df)
        self.features = list((set(self.features) - set(self.constant_features)))

        return self.transform(df)

    def transform(self, df):
        _df = df.copy()
        # _df = self.threshold(_df)
        _df = self.remove_constant(_df)

        _df.fillna(0, inplace=True)

        return _df


class ModelPipeline:
    def __init__(
        self,
        model: BaseEstimator,
        preprocessor: Preprocessor,
        label_encoder=LabelEncoder,
        scaler=None,
        label_name="is_breaking",
    ):
        self.model = model
        self.preprocessor = preprocessor
        self.label_name = label_name

        # if the class is passed, instantiate it
        # if the object is passed, use it
        if type(label_encoder) == type:
            self.label_encoder = label_encoder()
        elif isinstance(label_encoder, LabelEncoder):
            self.label_encoder = label_encoder
        else:
            self.label_encoder = None

        self.scaler = scaler if scaler else StandardScaler()

    def fit(self, df):
        _df = df.copy()
        _df = self.preprocessor.fit_transform(_df)

        X = _df[self.preprocessor.features].values
        y = _df[self.label_name].values

        if self.label_encoder:
            y = self.label_encoder.fit_transform(y)

        X = self.scaler.fit_transform(X)

        self.model.fit(X, y)

        return self

    def eval(self, df, split_by, resampler=SMOTE, random_state=None, verbose=False):
        _df = df.copy()
        _df = self.preprocessor.fit_transform(_df)

        batch_results = []

        for X_train, X_test, y_train, y_test, train_issues, test_issues in split_by(
            _df, self.preprocessor.features, random_state=random_state
        ):
            if verbose:
                print(
                    f"""
            Distribution:
                Training {X_train.shape}
                    | broken   (1): # {len(list(filter(lambda x: x == 1, y_train)))}
                    | fixed   (-1): # {len(list(filter(lambda x: x == -1, y_train)))}
                    | neutral  (0): # {len(list(filter(lambda x: x == 0, y_train)))}
                Validation {X_test.shape}
                    | broken   (1): # {len(list(filter(lambda x: x == 1, y_test)))}
                    | fixed   (-1): # {len(list(filter(lambda x: x == -1, y_test)))}
                    | neutral  (0): # {len(list(filter(lambda x: x == 0, y_test)))}
                """
                )

            if self.label_encoder:
                y_train = self.label_encoder.fit_transform(y_train)
                y_test = self.label_encoder.transform(y_test)

            if resampler:
                ros = resampler(random_state=random_state)
                X_train, y_train = ros.fit_resample(X_train, y_train)

            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)

            self.model.fit(X_train, y_train)

            y_pred = self.model.predict(X_test)

            if self.label_encoder:
                y_pred = self.label_encoder.inverse_transform(y_pred)
                y_test = self.label_encoder.inverse_transform(y_test)

            batch_results.append((y_test, y_pred, test_issues))

        return batch_results

    def predict(self, df):
        _df = df.copy()
        _df = self.preprocessor.transform(_df)

        X = _df[self.preprocessor.features].values

        X = self.scaler.transform(X)

        y_pred = self.model.predict(X)

        if self.label_encoder:
            y_pred = self.label_encoder.inverse_transform(y_pred)

        return y_pred


class WebpageClassifier(ABC):
    def predict(self, X: pd.DataFrame) -> List[int]:
        return self._predict(X)

    @abstractmethod
    def _predict(self, x: pd.DataFrame):
        raise NotImplementedError


class ThresholdHeuristicClassifier(WebpageClassifier):
    threshold: int

    def __init__(
        self, subtree_model: ModelPipeline, threshold, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.threshold = threshold
        self.subtree_model = subtree_model

    def _predict(self, x):
        true_labels = self.subtree_model.predict(x)
        true_labels = [max(0, x) for x in true_labels]
        if sum(true_labels) >= self.threshold:
            return 1

        return -1


class RatioHeuristicClassifier(WebpageClassifier):
    ratio: int

    def __init__(self, subtree_model: ModelPipeline, ratio, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ratio = ratio
        self.subtree_model = subtree_model

    def fit(X, Y):
        pass

    def _predict(self, x: pd.DataFrame):
        true_labels = self.subtree_model.predict(x)
        # only sum broken subtrees
        true_labels = [max(0, x) for x in true_labels]

        if sum(true_labels) / len(true_labels) >= self.ratio:
            return 1

        return -1
