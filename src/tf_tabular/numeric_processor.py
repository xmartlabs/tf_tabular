from .utils import build_projection_layer


class NumericProcessor:
    def __init__(
        self, num_projection: int | None = 32, l2_reg: float = 1e-5, cross_features: bool = True, builder=None
    ):
        self.num_projection = num_projection
        self.l2_reg = l2_reg
        self.cross_features = cross_features
        self.builder = builder

    def project(self, x):
        if self.builder is not None:
            return self.builder(x)
        if self.num_projection is None:
            return x
        return [build_projection_layer(x, self.num_projection, self.l2_reg, cross_features=self.cross_features)]
