from _typeshed import Incomplete

class Amalgamation:
    def actual_path(self, file_path): ...
    def find_included_file(self, file_path, source_dir): ...
    verbose: Incomplete
    prologue: Incomplete
    source_path: Incomplete
    included_files: Incomplete
    def __init__(self, args) -> None: ...
    def generate(self) -> None: ...

class TranslationUnit:
    cpp_comment_pattern: Incomplete
    c_comment_pattern: Incomplete
    string_pattern: Incomplete
    include_pattern: Incomplete
    pragma_once_pattern: Incomplete
    file_path: Incomplete
    file_dir: Incomplete
    amalgamation: Incomplete
    is_root: Incomplete
    content: Incomplete
    def __init__(self, file_path, amalgamation, is_root) -> None: ...

def main() -> None: ...
