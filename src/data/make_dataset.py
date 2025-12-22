import argparse
import numpy as np
import pandas as pd
from typing import Dict, Set, Tuple
from sklearn import model_selection, preprocessing
from pathlib import Path

from src.utils.constants import (
    DEFAULT_USER_COL as USER_COL,
    DEFAULT_ITEM_COL as ITEM_COL,
    DEFAULT_TARGET_COL as TARGET_COL,
    DEFAULT_TIMESTAMP_COL as TIMESTAMP_COL,
)

# --------------------------------------------------------------------------------------
# ----- Helper
# --------------------------------------------------------------------------------------


def clean_raw_dataframe(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, int, int]:
    """
    Fit user and item ID encoders on the full dataframe and remap IDs to contiguous
    indices starting from 0, suitable for nn.Embedding.

    The ID mapping (LabelEncoder / index mapping) must be done once on the full dataset,
    then split, and use that mapping everywhere.

    Parameters
    ----------
    df
        Raw interactions dataframe with at least columns:
        - USER_COL
        - ITEM_COL

    Returns
    -------
    df_encoded
        Copy of the input dataframe with USER_COL and ITEM_COL replaced by
        integer indices in [0, n_users) and [0, n_items), respectively.
    n_users
        Number of unique users (embedding size for user_embedding).
    n_items
        Number of unique items (embedding size for item_embedding).
    user_encoder
        Fitted LabelEncoder for user IDs (for future transform).
    item_encoder
        Fitted LabelEncoder for item IDs.
    """
    df_encoded = df.copy()

    user_encoder = preprocessing.LabelEncoder()
    item_encoder = preprocessing.LabelEncoder()

    df_encoded[USER_COL] = user_encoder.fit_transform(df_encoded[USER_COL].values)
    df_encoded[ITEM_COL] = item_encoder.fit_transform(df_encoded[ITEM_COL].values)

    n_users = len(user_encoder.classes_)
    n_items = len(item_encoder.classes_)

    return df_encoded, n_users, n_items


def add_negative_interactions(
    df: pd.DataFrame,
    user_positive_items: Dict[int, Set[int]],
    n_items: int,
    n_negatives: int,
    random_seed: int,
) -> pd.DataFrame:
    """
    Build an offline evaluation dataframe with fixed negatives.

    Parameters
    ----------
    df
        DataFrame with at least columns [USER, ITEM] representing held-out
        positives for each user.
    user_positive_items
        Dict[user] -> set(items) with ALL items the user has interacted with
        (train + val + test), used to avoid sampling true positives as negatives.
    n_items
        Total number of items (embedding size).
    n_negatives
        Number of negatives to sample per positive.
    random_seed
        Random seed for reproducible negative sampling.

    Returns
    -------
    df_with_negatives
        DataFrame with columns [USER, ITEM, TARGET].
    """
    rng = np.random.default_rng(random_seed)
    all_items = np.arange(n_items, dtype=np.int64)

    rows = []

    for row in df[[USER_COL, ITEM_COL]].itertuples(index=False):
        u = int(getattr(row, USER_COL))
        i_pos = int(getattr(row, ITEM_COL))

        # positive
        rows.append((u, i_pos, 1.0))

        pos_items = user_positive_items.get(u, set())
        if len(pos_items) >= n_items:
            # Degenerate case: user interacted with all items
            continue

        # candidate negatives = all items user has never interacted with
        # (convert set -> array once per row; you can optimise per user later)
        pos_arr = (
            np.fromiter(pos_items, dtype=np.int64)
            if pos_items
            else np.array([], dtype=np.int64)
        )
        candidates = np.setdiff1d(all_items, pos_arr, assume_unique=True)

        k = min(n_negatives, len(candidates))
        if k == 0:
            continue

        neg_items = rng.choice(candidates, size=k, replace=False)
        for j in neg_items:
            rows.append((u, int(j), 0.0))

    df_with_negatives = pd.DataFrame(rows, columns=[USER_COL, ITEM_COL, TARGET_COL])

    return df_with_negatives


# --------------------------------------------------------------------------------------
# ----- Main
# --------------------------------------------------------------------------------------


def ml_latest_small_user_item_interactions(
    input_data_dir: str,
    output_data_dir: str,
    val_split: float,
    test_split: float,
    n_negatives: int,
    random_seed: int,
) -> Tuple[int, int]:
    """
    Prepare implicit user-item interaction splits for ml-latest-small.

    Steps:
    - Load ratings parquet for ml-latest-small.
    - Rename columns to internal names (USER_COL, ITEM_COL, TIMESTAMP_COL, TARGET_COL).
    - Convert explicit ratings to implicit feedback (TARGET_COL = 1.0 for all interactions).
    - Encode user_id and item_id to contiguous indices [0, n_users), [0, n_items).
    - Randomly split into train / val / test on interactions (not stratified by label,
      since labels are all 1.0 for positives).
    - Optionally add offline negative interactions to val and test via
      `add_negative_interactions`.
    - Save:
        full.parquet      : all encoded interactions (positives only)
        train.parquet     : training positives
        val.parquet       : validation positives (+ negatives if n_negatives > 0)
        train_val.parquet : train + val positives (no negatives)
        test.parquet      : test positives (+ negatives if n_negatives > 0)

    Parameters
    ----------
    input_data_dir
        Input directory in which to save the processed parquet files.
    output_data_dir
        Output directory in which to save the processed parquet files.
    val_split
        Fraction of the data (of the remaining after test_split) to allocate to validation.
        For example, val_split=0.1, test_split=0.2 → 20% test, 10% of original as val.
    test_split
        Fraction of the full dataset to allocate to test.
    n_negatives
        Number of offline negatives per positive to sample for val and test.
        If 0, no negatives are added (only positives are saved).
    random_seed
        Random seed for reproducible splits and negative sampling.

    Returns
    -------
    n_users
        Total number of unique encoded users in the dataset.
    n_items
        Total number of unique encoded items in the dataset.
    """

    input_data_dir = Path(output_data_dir)
    output_data_dir_path = Path(output_data_dir)
    output_data_dir_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_data_dir / "ratings.parquet")

    df.rename(
        columns={
            "userId": USER_COL,
            "movieId": ITEM_COL,
            "timestamp": TIMESTAMP_COL,
            "rating": TARGET_COL,
        },
        inplace=True,
    )

    # Explicit -> implicit: everything in this file is a positive interaction
    df[TARGET_COL] = 1.0

    # Encode user and item IDs **on the full dataframe**
    df, n_users, n_items = clean_raw_dataframe(df)

    # ----------------------------------------------------------------------------------
    # ----- Splits (cannot stratify on TARGET_COL because all labels are 1.0)
    # ----- If later timebased: drop SHUFFLE = True
    # ----------------------------------------------------------------------------------
    df_train_val, df_test = model_selection.train_test_split(
        df,
        test_size=test_split,
        random_state=random_seed,
        shuffle=True,
        stratify=None,
    )

    # Adjust val ratio relative to the remaining data
    relative_val_size = val_split / (1.0 - test_split)

    df_train, df_val = model_selection.train_test_split(
        df_train_val,
        test_size=relative_val_size,
        random_state=random_seed,
        shuffle=True,
        stratify=None,
    )

    # ----------------------------------------------------------------------------------
    # ----- Offline negatives for validation and test
    # ----------------------------------------------------------------------------------
    if n_negatives > 0:
        # build global user -> positive items mask from full df
        user_positive_items = df.groupby(USER_COL)[ITEM_COL].apply(set).to_dict()
        n_items_total = df[ITEM_COL].max() + 1  # should equal n_items

        df_val = add_negative_interactions(
            df=df_val,
            user_positive_items=user_positive_items,
            n_items=n_items_total,
            n_negatives=n_negatives,
            random_seed=random_seed,
        )

        df_test = add_negative_interactions(
            df=df_test,
            user_positive_items=user_positive_items,
            n_items=n_items_total,
            n_negatives=n_negatives,
            random_seed=random_seed,
        )

    print(
        f"Sizes -> train: {len(df_train)}, "
        f"val: {len(df_val)}, "
        f"train_val: {len(df_train_val)}, "
        f"test: {len(df_test)}, "
        f"full: {len(df)}"
    )

    # -------------------------------------------------------------------------
    # Persist splits
    # -------------------------------------------------------------------------
    df.to_parquet(output_data_dir_path / "full.parquet", index=False)
    df_train.to_parquet(output_data_dir_path / "train.parquet", index=False)
    df_val.to_parquet(output_data_dir_path / "val.parquet", index=False)
    df_train_val.to_parquet(output_data_dir_path / "train_val.parquet", index=False)
    df_test.to_parquet(output_data_dir_path / "test.parquet", index=False)

    return n_users, n_items


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Compute dataset gold version")

    # define the --dataset argument
    parser.add_argument(
        "--dataset", type=str, choices=["ml_latest_small_user_item_interactions"]
    )

    # define the --val_split argument
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
    )

    # define the --test_split argument
    parser.add_argument(
        "--test_split",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--n_negatives",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    globals()[args.dataset](
        val_split=args.val_split,
        test_split=args.test_split,
        n_negatives=args.n_negatives,
        random_seed=args.random_seed,
    )
