from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from BreakageClassifier.code.features import constants


class Dataset:
    name: str
    dir_name: Path
    features_filename: str = "features.csv"
    label_col: str = "is_breaking"
    features: list = constants.FEATURES

    def __init__(
        self,
        name: str,
        dir_name: str,
        features_filename: str = "features.csv",
        label_col: str = "is_breaking",
        features: list = constants.FEATURES,
        is_constructed: bool = False,
        remove_empty_issues: bool = False,
    ):
        self.name = name
        if dir_name is None:
            self.dir_name = None
        else:
            self.dir_name = Path(dir_name).resolve()
        self.features_filename = features_filename
        self.label_col = label_col
        self.features = features
        self.is_constructed = is_constructed
        self.remove_empty_issues = remove_empty_issues

        self._raw = None
        self._data = None

    @property
    def features_fp(self):
        if self.is_constructed:
            raise ValueError(
                "Cannot access features_fp if it is constructed from other datasets"
            )
        return self.dir_name / self.features_filename

    @property
    def raw(self) -> pd.DataFrame:
        if self._raw is None and not self.is_constructed:
            self._raw = pd.read_csv(self.features_fp)

            if self.remove_empty_issues:
                self._raw = self._raw[~self._raw["issue"].isin(self.issues_to_remove())].copy()

        return self._raw
    
    def _data_from_raw(self):
        wanted_cols = set(self.features + [self.label_col] + ["issue"])
        existing_cols = set(self.raw.columns.tolist())

        if not wanted_cols.issubset(existing_cols):
            print(f"WARNING: Missing columns: {wanted_cols - existing_cols}")

        self._data = self.raw[list(wanted_cols.intersection(existing_cols))]
        self.features = list(wanted_cols & existing_cols)
    

    @property
    def data(self) -> pd.DataFrame:
        if self._data is None and not self.is_constructed:
            self._data_from_raw()
        return self._data

    def issues_to_remove(self):
        issue_to_remove = []

        for issue in self._raw["issue"].unique():
            breaking_counts = self._raw[self._raw["issue"] == issue][
                "is_breaking"
            ].value_counts()

            if 1 not in breaking_counts:
                issue_to_remove.append(issue)

            if -1 in breaking_counts and breaking_counts[-1] > 30:
                issue_to_remove.append(issue)

            if 1 in breaking_counts and breaking_counts[1] > 30:
                issue_to_remove.append(issue)

        return issue_to_remove

    def rename(self, name: str):
        self.name = name
        return self

    def __repr__(self):
        return f"Dataset(name={self.name}, dir_name={self.dir_name}, n_rows={len(self.data)}, n_features={len(self.features)})"

    def __add__(self, other):
        
       
        if not isinstance(other, Dataset):
            raise TypeError("Can only add Datasets together")
        
        # load data if not already loaded
        self.data
        other.data
        

        if set(self.features) != set(other.features):
            raise ValueError("Can only add Datasets with the same features")

        if self.label_col != other.label_col:
            raise ValueError("Can only add Datasets with the same label column")

        if self.dir_name == other.dir_name:
            raise ValueError("Can only add Datasets with different directory names")

        d = Dataset(
            name=f"{self.name}+{other.name}",
            dir_name=None,
            features_filename=None,
            label_col=self.label_col,
            features=self.features,
            is_constructed=True,
        )

        # align columns
        self_raw_cols = set(self.raw.columns.tolist())
        other_raw_cols = set(other.raw.columns.tolist())
        
        self_cols = set(self.data.columns.tolist())
        other_cols = set(other.data.columns.tolist())

        if not self_cols.issubset(other_cols):
            print(f"WARNING: Missing columns: {self_cols - other_cols}")

        if not other_cols.issubset(self_cols):
            print(f"WARNING: Missing columns: {other_cols - self_cols}")
            
        same_cols = self_cols.intersection(other_cols)

        self_data = self.data[list(same_cols)]
        other_data = other.data[list(same_cols)]

        same_raw_cols = self_raw_cols.intersection(other_raw_cols)

        self_raw = self.raw[list(same_raw_cols)].copy()
        other_raw = other.raw[list(same_raw_cols)].copy()
        
        self_raw['dataset'] = self.name
        other_raw['dataset'] = other.name

        d._data = pd.concat([self_data, other_data], ignore_index=True)
        d._raw = pd.concat([self_raw, other_raw], ignore_index=True)

        return d
    
    def drop_issues(self, issues):
        self._raw = self.raw[~self.raw['issue'].isin(issues)]
        self._data_from_raw()
        
        return self


class Datasets:
    # can query like dict by name
    # can iterate over values
    # immutable

    _list: List[Dataset]
    _dict: Dict[str, Dataset]

    def __init__(self, *datasets: List[Dataset]):
        if len(datasets) == 0:
            raise ValueError("Must provide at least one dataset")

        self._list: List[Dataset] = datasets
        self._dict = {d.name: d for d in datasets}
        self._all = None

    def __getattribute__(self, __name: str) -> Any:
        try:
            return super().__getattribute__(__name)
        except AttributeError as e:
            if __name in self._dict:
                return self._dict[__name]
            else:
                raise e

    def __iter__(self):
        
        for d in self._list:
            yield d
        
        yield self.all

    @property
    def features(self):
        return self._list[0].data.columns.tolist()

    @property
    def all(self):
        if self._all is None:
            self._all = self._list[0]
            for d in self._list[1:]:
                self._all += d

            self._all.rename("all")

        return self._all
