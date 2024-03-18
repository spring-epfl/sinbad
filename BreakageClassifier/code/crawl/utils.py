from pathlib import Path
import time
from typing import Any, Callable, List, Optional


def get_ignored_issues(ignore: Optional[Path]) -> List[int]:
    ignored = []

    if ignore:
        with open(str(ignore.resolve()), "r") as f:
            ignored = f.readlines()
        ignored = [int(x) for x in ignored]

    return ignored


def wait_for_val(getter: Callable, block_val = None) -> Any:
    """wait until value is not None"""

    while True:
        val = getter()
        if val != block_val:
            return val
        time.sleep(0.1)
