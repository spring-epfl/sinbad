from .types import BrowserContext, BlockVo


def _compute_relative_measures(block: BlockVo, browser_context: BrowserContext):

    cx, cy = block.x + block.width / 2, block.y + block.height / 2

    out = {
        "center_x": cx / browser_context.window_size["width"],
        "center_y": cy / browser_context.window_size["height"],
        "top_left_x": block.x / browser_context.window_size["width"],
        "top_left_y": block.y / browser_context.window_size["height"],
        "bottom_right_x": (block.x + block.width)
        / browser_context.window_size["width"],
        "bottom_right_y": (block.y + block.height)
        / browser_context.window_size["height"],
        "width": block.width / browser_context.window_size["width"],
        "height": block.height / browser_context.window_size["height"],
        "area": block.width
        * block.height
        / (
            browser_context.window_size["height"] * browser_context.window_size["width"]
        ),
    }

    out["is_visible"] = (0 <= block.x <= browser_context.window_size["width"]) and (
        0 <= block.x <= browser_context.window_size["height"]
    )

    return out
