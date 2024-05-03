from typing import Dict, Text

import tensorflow as tf
import tensorflow_recommenders as tfrs


class MovielensModel(tfrs.Model):
    def __init__(self, user_model: tf.keras.Model, movie_model: tf.keras.Model):
        super().__init__()
        self.movie_model = movie_model
        self.user_model = user_model

    def prepare_task(self, movies):
        id_candidates = (
            movies.ragged_batch(1024)
            .prefetch(tf.data.AUTOTUNE)
            .cache()
            .map(lambda movie: (movie["movie_title"], self.movie_model(movie)))
        )

        metrics = tfrs.metrics.FactorizedTopK(
            candidates=tfrs.layers.factorized_top_k.Streaming(k=100).index_from_dataset(id_candidates),
            ks=[1, 5, 100],
            name="factk",
        )
        loss = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
        )
        task = tfrs.tasks.Retrieval(metrics=metrics, remove_accidental_hits=True, loss=loss)
        self.task = task

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        # We pick out the user features and pass them into the user model.
        user_embeddings = self.user_model(features["user_id"])
        # And pick out the movie features and pass them into the movie model,
        # getting embeddings back.
        positive_movie_embeddings = self.movie_model(
            {"movie_title": features["movie_title"], "movie_genres": features["movie_genres"]}
        )

        # The task computes the loss and the metrics.
        return self.task(
            user_embeddings,
            positive_movie_embeddings,
            candidate_ids=features["movie_title"],
            candidate_sampling_probability=features["sampling_prob"],
        )
