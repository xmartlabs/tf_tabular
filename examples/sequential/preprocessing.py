import pandas as pd
import numpy as np
import tensorflow as tf

from tf_tabular.utils import get_vocab


def divide_ratings_by_mean_user_rating(ratings: pd.DataFrame, user_id_column="user_id"):
    mean_ratings = ratings.groupby([user_id_column])[["user_rating"]].agg("mean").reset_index()
    mean_ratings = mean_ratings.rename(columns={"user_rating": "mean_rating"})
    ratings = ratings.merge(mean_ratings, on=user_id_column)
    ratings["user_rating"] = ratings["user_rating"] - ratings["mean_rating"]
    return ratings


def split_by_user(
    ratings: pd.DataFrame,
    max_y_cutoff,
):
    ratings = ratings.sort_values(["user_id", "timestamp"])
    ratings = divide_ratings_by_mean_user_rating(ratings)

    unique_users = ratings["user_id"].unique()
    ratings = ratings[["user_id", "movie_id", "user_rating"]].groupby(["user_id"], as_index=False).agg(list)

    def cutoff(x):
        return min(int(len(x) * 0.2), max_y_cutoff)

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
    print(f"Unique users: {num_users}")
    val_users = unique_users[: int(num_users * 0.2)]
    train_users = unique_users[int(num_users * 0.2) :]
    train_set = ratings[ratings.user_id.isin(train_users)]
    val_set = ratings[ratings.user_id.isin(val_users)]

    print(f"Train set size: {train_set.shape}")
    print(f"Validation set size: {val_set.shape}")
    return train_set, val_set


def join_movie_info(ratings, movies_df):
    ratings = ratings.merge(movies_df, left_on="target_id", right_on="movie_id", how="left")
    ratings = ratings.drop("movie_id", axis=1)
    return ratings


def load_tf_dataset_from_pandas(df, ragged_columns, other_columns):
    df_dict = {}
    for rc in ragged_columns:
        tensor = tf.ragged.constant(df[rc])
        df_dict[rc] = tensor
    df = df[other_columns]
    df_dict.update(dict(df))
    ds = tf.data.Dataset.from_tensor_slices(df_dict)
    ds = ds.shuffle(1000)
    return ds


def compute_sampling_probabilities(df, column, sp_name="sampling_prob"):
    vc = df[column].value_counts(normalize=True)
    df[sp_name] = df[column].map(lambda x: vc[x]).astype(np.float32)
    return df


def build_vocabs(df, cols):
    vocabs = {}
    for column in cols:
        vocabs[column] = get_vocab(df[column])
    return vocabs


def preprocess_dataset(ratings_df, movies_df):
    train_df, val_df = split_by_user(ratings_df, max_y_cutoff=5)
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
