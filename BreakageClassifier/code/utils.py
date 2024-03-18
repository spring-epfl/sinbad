import traceback
from pathlib import Path
from .logger import LOGGER


def graph_node_id(browser_id, visit_id, node_id):
    try:
        return f"{int(browser_id)}={int(visit_id)}={node_id}"
    except ValueError as v:
        return None


def return_none_if_fail(is_debug=False):
    def _return_none_if_fail(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if is_debug:
                    LOGGER.exception(f"func {func} failed: {e}", exc_info=True)
                return None

        return wrapper

    return _return_none_if_fail


def path_from_home(path):
    return path.replace("~", str(Path.home()))
