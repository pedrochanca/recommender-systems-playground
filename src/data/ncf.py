import pandas as pd
from torch.utils.data import DataLoader

from src.data.datasets import PointwiseImplicitDataset, OfflineImplicitDataset
from src.data.samplers import GlobalUniformNegativeSampler
from src.utils.constants import (
    DEFAULT_USER_COL as USER_COL,
    DEFAULT_ITEM_COL as ITEM_COL,
    DEFAULT_TARGET_COL as TARGET_COL,
    DEFAULT_TIMESTAMP_COL as TIMESTAMP_COL,
)


class NCFDataset:

    def __init__(
        self,
        train_file_path: str,
        test_file_path: str,
        full_file_path: str,
        n_negatives: int = 4,
        user_col: str = USER_COL,
        item_col: str = ITEM_COL,
        timestamp_col: str = TIMESTAMP_COL,
        target_col: str = TARGET_COL,
    ):

        self.train_file_path = train_file_path
        self.test_file_path = test_file_path
        self.n_negatives = n_negatives

        self.user_col = user_col
        self.item_col = item_col
        self.timestamp_col = timestamp_col
        self.target_col = target_col

        self.df_train = pd.read_parquet(train_file_path)
        self.df_test = pd.read_parquet(test_file_path)
        self.df_full = pd.read_parquet(full_file_path)

        self.n_users = self.df_full[self.user_col].max() + 1
        self.n_items = self.df_full[self.item_col].max() + 1

        self.train_set = self.train_dataset()
        self.test_set = self.test_dataset()

    def train_dataset(self):

        self.user_positive_items = (
            self.df_full.groupby(self.user_col)[self.item_col].apply(set).to_dict()
        )

        self.negative_sampler = GlobalUniformNegativeSampler(
            self.n_items, self.user_positive_items
        )

        return PointwiseImplicitDataset(
            users=self.df_train[self.user_col].values,
            items=self.df_train[self.item_col].values,
            timestamps=self.df_train[self.timestamp_col].values,
            negative_sampler=self.negative_sampler,
            n_negatives=self.n_negatives,
        )

    def test_dataset(self):

        return OfflineImplicitDataset(
            users=self.df_test[self.user_col].values,
            items=self.df_test[self.item_col].values,
            targets=self.df_test[self.target_col].values,
        )

    def train_loader(self, batch_size: int, n_workers: int, shuffle: bool = True):
        return DataLoader(
            self.train_set,
            batch_size=batch_size,
            num_workers=n_workers,
            shuffle=shuffle,
        )

    def test_loader(self, batch_size: int, n_workers: int, shuffle: bool = False):
        return DataLoader(
            self.test_set,
            batch_size=batch_size,
            num_workers=n_workers,
            shuffle=shuffle,
        )
