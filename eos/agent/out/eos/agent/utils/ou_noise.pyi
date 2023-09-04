from _typeshed import Incomplete

class OUActionNoise:
    theta: Incomplete
    mean: Incomplete
    std_dev: Incomplete
    dt: Incomplete
    x_initial: Incomplete
    def __init__(
        self,
        mean,
        std_deviation,
        theta: float = ...,
        dt: float = ...,
        x_initial: Incomplete | None = ...,
    ) -> None: ...
    x_prev: Incomplete
    def __call__(self): ...
    def reset(self) -> None: ...
