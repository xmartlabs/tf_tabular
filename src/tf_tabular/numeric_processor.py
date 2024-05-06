from typing import List
import tensorflow as tf

from .utils import build_projection_layer


class NumericProcessor:
    """This class is used to project numerical features to a lower or higher dimension."""

    def __init__(
        self,
        num_projection: int | None = None,
        l2_reg: float = 1e-5,
        cross_features: bool = True,
        projection_activation: str = "relu",
        builder=None,
    ):
        self.num_projection = num_projection
        self.l2_reg = l2_reg
        self.cross_features = cross_features
        self.builder = builder
        self.projection_activation = projection_activation

    def project(self, x: List[tf.Tensor]) -> List[tf.Tensor]:
        """If num_projection is not None, project the numerical features to a lower or higher dimension.
        If a builder is provided, use that to build the projection layer. Otherwise, use the default projection layer.
        """
        if self.builder is not None:
            return self.builder(x)
        if self.num_projection is None:
            return x
        return [
            build_projection_layer(
                x,
                self.num_projection,
                self.l2_reg,
                activation=self.projection_activation,
                cross_features=self.cross_features,
            )
        ]
