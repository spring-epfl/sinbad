from __future__ import annotations
from math import exp, log
from .types import BrowserContext, PopulationLayout, BlockVo
from .utils import _compute_relative_measures


def _get_structural_score(pos: dict, population_layout: PopulationLayout):

    # edge case if element not in viewport score it as 0
    if (
        pos["top_left_x"] >= 1
        or pos["bottom_right_x"] <= 0
        or pos["top_left_y"] >= 1
        or pos["bottom_right_y"] < 0
        or pos["width"] == 0
        or pos["height"] == 0
    ):
        return 0

    # selection boundries
    # we want elements within a specific size boundries
    # if pos["area"] < 0.01 or pos["area"] > 0.2:
    #     return 0

    # some hyperparameters
    ccf = 10  # centrality coefficient
    slf = 1.3  # size log factor

    # check centrality on screen
    # centrality: function of cartesian distance between center coordinates and (0.5, 0.5) : middle of screen
    centrality = exp(-ccf * (pos["center_x"] - 0.5) ** 2 + (pos["center_y"] - 0.5))

    # check the leftiness
    # leftiness: function of distance between left boundary and element
    # leftiness also depends if the element is visible or not
    leftiness = exp(-ccf * pos["center_x"] ** 2)

    ## check the size
    # size: function of the area, width and height

    aspect = (
        pos["height"] / pos["width"]
        if pos["height"] < pos["width"]
        else pos["width"] / pos["height"]
    )

    size = max(pos["area"] * slf * log(100 * aspect), 0)

    ## check closeness to population

    closeness = exp(
        -ccf * (population_layout.center_x - 0.5) ** 2
        + (population_layout.center_y - 0.5)
    )

    ## combining the structural measures
    return (centrality + leftiness + closeness) * size


def _is_visible(node):
    return node.visual_cues["is_visible"]


def saliency_score(
    block: BlockVo, population_layout: PopulationLayout, browser_context: BrowserContext
):

    # trivial edge cases
    if block.width == 0 and block.height == 0:
        return 0

    scores = []
    weights = []

    # check visibility
    if all([not _is_visible(node) for node in block.boxs]):
        return 0

    # block parameters
    relative_pos = _compute_relative_measures(block, browser_context)

    # positional heuristics
    positional_weight = 0.4
    positional_score = _get_structural_score(relative_pos, population_layout)
    scores.append(positional_score)
    weights.append(positional_weight)

    # tag heuristics
    tag_weight = 0.4

    return sum([x * w for x, w in zip(scores, weights)]) / sum(weights)
