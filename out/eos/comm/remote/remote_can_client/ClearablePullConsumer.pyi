from collections.abc import Generator

from _typeshed import Incomplete
from rocketmq.client import PullConsumer

class ClearablePullConsumer(PullConsumer):
    def get_message_queue_offset(self, mq): ...
    def set_message_queue_offset(self, mq, offset) -> None: ...
    def clear_history(self, topic, expression: str = ...) -> None: ...
    def pull(
        self, topic, expression: str = ..., max_num: int = ...
    ) -> Generator[Incomplete, None, None]: ...
