from typing import List
import logging

import pandas as pd
import numpy as np
import tensorflow as tf

from tf_tabular.utils import get_vocab


logger = logging.getLogger(__name__)


def normalize_ratings_by_mean_user_rating(ratings: pd.DataFrame, user_id_column="user_id"):
    """Normalizes the ratings by subtracting the mean rating on a user basis.

    :param pd.DataFrame ratings: User ratings dataset
    :param str user_id_column: Name of the column of the user id, defaults to "user_id"
    :return pd.DataFrame: Updated dataframe
    """
    mean_ratings = ratings.groupby([user_id_column])[["user_rating"]].agg("mean").reset_index()
    mean_ratings = mean_ratings.rename(columns={"user_rating": "mean_rating"})
    ratings = ratings.merge(mean_ratings, on=user_id_column)
    ratings["user_rating"] = ratings["user_rating"] - ratings["mean_rating"]
    return ratings


def split_by_user(
    ratings: pd.DataFrame,
    max_y_cutoff: int,
    val_split: float = 0.2,
    target_split: float = 0.2,
):
    """Split dataset by users.

    :param pd.DataFrame ratings: User ratings dataframe
    :param int max_y_cutoff: Max number of movies that will be used as targets per user
    :param float val_split: Validation dataset split, defaults to 0.2
    :param float target_split: Percent of user actions to leave as prediction target, defaults to 0.2
    :return tuple (pd.DataFrame, pd.DataFrame): Train and validation datasets
    """
    ratings = ratings.sort_values(["user_id", "timestamp"])
    ratings = normalize_ratings_by_mean_user_rating(ratings)

    unique_users = ratings["user_id"].unique()
    ratings = ratings[["user_id", "movie_id", "user_rating"]].groupby(["user_id"], as_index=False).agg(list)

    def cutoff(x):
        return min(int(len(x) * target_split), max_y_cutoff)

    ratings["user_history"] = ratings["movie_id"].apply(lambda x: x[: -cutoff(x)])
    ratings["target_id"] = ratings["movie_id"].apply(lambda x: x[-cutoff(x) :])
    ratings["history_ratings"] = ratings["user_rating"].apply(lambda x: x[: -cutoff(x)])
    ratings["target_rating"] = ratings["user_rating"].apply(lambda x: x[-cutoff(x) :])
    ratings = ratings.drop("movie_id", axis=1).drop("user_rating", axis=1)

    ratings = ratings.explode(["target_id", "target_rating"]).reset_index()
    ratings = ratings.drop("target_rating", axis=1)

    ratings = ratings.dropna(subset=["target_id"])

    np.random.shuffle(unique_users)
    num_users = len(unique_users)
    logger.info(f"Unique users: {num_users}")
    val_users = unique_users[: int(num_users * val_split)]
    train_users = unique_users[int(num_users * val_split) :]
    train_set = ratings[ratings.user_id.isin(train_users)]
    val_set = ratings[ratings.user_id.isin(val_users)]

    logger.info(f"Train set size: {train_set.shape}")
    logger.info(f"Validation set size: {val_set.shape}")
    return train_set, val_set


def join_movie_info(ratings: pd.DataFrame, movies_df: pd.DataFrame):
    """Merge ratings and movies dataframes.

    :param pd.DataFrame ratings: Contains the ratings data
    :param pd.DataFrame movies_df: Contains movie metadata such as title and genres
    :return pd.DataFrame: Joined dataframe
    """
    ratings = ratings.merge(movies_df, left_on="target_id", right_on="movie_id", how="left")
    ratings = ratings.drop("movie_id", axis=1)
    return ratings


def load_tf_dataset_from_pandas(df: pd.DataFrame, ragged_columns: List[str], other_columns: List[str]):
    """Create a tf.Dataset from a Pandas DataFrame

    :param pd.DataFrame df: Dataframe to convert
    :param List[str] ragged_columns: List of ragged columns in df
    :param List[str] other_columns: List of other columns in df that should be included in the dataset
    :return tf.Dataset: tf Dataset containing tthe columns in ragged_columns and other_columns
    """
    df_dict = {}
    for rc in ragged_columns:
        tensor = tf.ragged.constant(df[rc])
        df_dict[rc] = tensor
    df = df[other_columns]
    df_dict.update(dict(df))
    ds = tf.data.Dataset.from_tensor_slices(df_dict)
    ds = ds.shuffle(1000)
    return ds


def compute_sampling_probabilities(df: pd.DataFrame, column: str, sp_name: str = "sampling_prob"):
    """Compute the sampling probability for each value in a column.

    :param pd.DataFrame df: Dataframe containing the data
    :param str column: Column for which to compute SP
    :param str sp_name: Name of new column, defaults to "sampling_prob"
    :return pd.DataFrame: Updated dataframe
    """
    vc = df[column].value_counts(normalize=True)
    df[sp_name] = df[column].map(lambda x: vc[x]).astype(np.float32)
    return df


def build_vocabs(df: pd.DataFrame, cols: List[str]):
    """Build vocabularies for categorical columns.

    :param pd.DataFrame df: Dataframe containing the training data
    :param List[str] cols: List of columns for which to compute vocabularies
    :return dict: Dictionary mapping column name to vocabulary
    """
    vocabs = {}
    for column in cols:
        vocabs[column] = get_vocab(df[column])
    return vocabs


def preprocess_dataset(ratings_df: pd.DataFrame, movies_df: pd.DataFrame, max_y_cutoff: int = 5):
    """Preprocess the dataset for training.

    :param pd.DataFrame ratings_df: Dataframe containing Movielens ratings
    :param pd.DataFrame movies_df: Dataframe containing Movielens movies
    :param int max_y_cutoff: Max number of movies that will be used as targets per user, defaults to 5
    :return tuple: Returns a tuple of train dataset (tf.Dataset), validation dataset (tf.Dataset) and vocabs (dict)
    """
    train_df, val_df = split_by_user(ratings_df, max_y_cutoff=max_y_cutoff)
    train_df = join_movie_info(train_df, movies_df)
    val_df = join_movie_info(val_df, movies_df)
    vocabs = build_vocabs(train_df, ["user_history", "movie_genres", "target_id"])
    train_df = compute_sampling_probabilities(train_df, "target_id")
    val_df = compute_sampling_probabilities(val_df, "target_id")

    ragged_cols = ["user_history", "history_ratings", "movie_genres"]
    simple_cols = ["movie_title", "target_id", "sampling_prob"]
    train_ds = load_tf_dataset_from_pandas(train_df, ragged_cols, simple_cols)
    val_ds = load_tf_dataset_from_pandas(val_df, ragged_cols, simple_cols)

    return train_ds, val_ds, vocabs
