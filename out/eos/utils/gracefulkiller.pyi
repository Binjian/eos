class GracefulKiller:
    kill_now: bool
    def __init__(self) -> None: ...
    def exit_gracefully(self, signum, frame) -> None: ...
