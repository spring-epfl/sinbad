from __future__ import annotations
import base64
from hashlib import md5
from io import BytesIO
import json
import os
from typing import Callable, List, Tuple
from matplotlib import patches, pyplot as plt
from matplotlib.axes import Axes

from Saliency.segment.vips.block_vo import BlockVo
from Saliency.segment.vips.dom_node import DomNode
from math import exp, log
from abc import ABC, abstractmethod
from typing import List
from PIL import Image


class DomNode:
    """A representation of a HTML DOM node"""

    __slots__ = (
        "id",
        "nodeType",
        "tagName",
        "nodeName",
        "nodeValue",
        "visual_cues",
        "attributes",
        "childNodes",
        "parentNode",
        "t_enter",
        "t_leave",
    )

    def __init__(self, node_type: int, id: int, t_enter: int):
        self.nodeType = node_type
        self.attributes = dict()
        self.childNodes = []
        self.visual_cues = dict()
        self.id = id
        self.t_enter = t_enter

    def create_element(self, tagName: str):
        """creates an element based on the tag name

        Args:
            tagName (str): the tag type of the DOM node
        """
        self.nodeName = tagName
        self.tagName = tagName

    def create_text_node(self, node_value: str, parent_node: DomNode):
        """create a text node

        Args:
            node_value (str): the text value of the node
            parent_node (DomNode): the parent DomNode
        """
        self.nodeName = "#text"
        self.nodeValue = node_value
        self.parentNode = parent_node

    def create_comment(self, node_value: str, parent_node: DomNode):
        """create a comment node

        Args:
            node_value (str): the text value of the node
            parent_node (DomNode): the parent DomNode
        """
        self.nodeName = "#comment"
        self.nodeValue = node_value
        self.parentNode = parent_node

    def set_attributes(self, attributes: list[dict]):
        """set the attributes of node

        Args:
            attribute (list[dict]): the list of attributes of the HTML element
        """
        self.attributes = attributes

    def set_visual_cues(self, visual_cues: dict):
        """set the visual cues of node

        Args:
            visual_cues (dict): a dictionary of the visual and css attributes of node
        """
        self.visual_cues = visual_cues

    def append_child(self, child_node: DomNode):
        """add a child to the node

        Args:
            child_node (DomNode)
        """
        self.childNodes.append(child_node)
        child_node.parentNode = self

    def leave(self, t: int):
        """set the t_leave of node

        Args:
            t (int)
        """
        self.t_leave = t


class BlockVo:
    id = None
    x = 0
    y = 0
    width = 0
    height = 0
    boxs = []
    parent = None
    children = []
    isVisualBlock = True
    isDividable = True
    Doc = 0
    count = 1


class BrowserContext:
    window_size = {"width": 0, "height": 0}

    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            self.__setattr__(key, val)


class PopulationLayout:
    center_x = 0
    center_y = 0

    def __init__(self, **kwargs) -> None:
        for key, val in kwargs.items():
            self.__setattr__(key, val)

    @staticmethod
    def from_block_list(
        block_list: List[BlockVo], browser_context: BrowserContext
    ) -> PopulationLayout:
        """Generates a population layout from a list of blocks and a browser context

        Args:
            block_list (List[BlockVo]): the list of blocks
            browser_context (BrowserContext): the browser context

        Returns:
            PopulationLayout: the population layout
        """
        comp = [_compute_relative_measures(x, browser_context) for x in block_list]
        comp_x = [x["center_x"] * x["area"] for x in comp]
        comp_y = [x["center_y"] * x["area"] for x in comp]
        comp_area = [x["area"] for x in comp]

        if sum(comp_area) == 0:
            return (0.5, 0.5)

        return PopulationLayout(
            center_x=sum(comp_x) / sum(comp_area), center_y=sum(comp_x) / sum(comp_area)
        )


class SegmentationGenerator(ABC):
    def __init__(self, node_list: List[DomNode], browser_context: BrowserContext):
        self.node_list = node_list
        self.browser_context = browser_context

    @abstractmethod
    def service(self) -> List[BlockVo]:
        """segments a web document represented by self.node_list under a given BrowserContext

        Returns:
            List[BlockVo]: the list of blocks
        """
        raise NotImplementedError()

    def setRound(self, i: int):
        """Set the total number of iterations the generator should repeat

        Args:
            i (int): rounds number
        """

        self.Round = i


class DomEncoder:
    """Based from PythonVispy implementation. Encodes the dom with visual attributes"""

    count = 0
    imgOut = None
    html = None
    cssBoxList = dict()
    nodeList = []
    count3 = 0
    t = 0

    def __init__(self, logger):
        self.logger = logger

    def to_dom(self, obj, parent_node=None):
        """Converts the json node tree to dom"""
        if isinstance(obj, str):
            json_obj = json.loads(obj)  # use json lib to load our json string
        else:
            json_obj = obj
        node_type = json_obj["nodeType"]
        node = DomNode(node_type, self.count, self.t)
        self.count += 1
        self.t += 1

        if node_type == 1:  # ELEMENT NODE
            node.create_element(json_obj["tagName"])
            attributes = json_obj["attributes"]
            if attributes is not None:
                node.set_attributes(attributes)
            visual_cues = json_obj["visual_cues"]
            if visual_cues is not None:
                node.set_visual_cues(visual_cues)
        elif node_type == 3:
            node.create_text_node(json_obj["nodeValue"], parent_node)
            if node.parentNode is not None:
                visual_cues = node.parentNode.visual_cues
                if visual_cues is not None:
                    node.set_visual_cues(visual_cues)
        else:
            return node

        self.nodeList.append(node)
        if node_type == 1:
            child_nodes = json_obj["childNodes"]
            for _, child_node in enumerate(child_nodes):
                if child_node["nodeType"] == 1:
                    node.append_child(self.to_dom(child_node, node))
                if child_node["nodeType"] == 3:
                    try:
                        if not child_node["nodeValue"].isspace():
                            node.append_child(self.to_dom(child_node, node))
                    except KeyError:
                        self.logger.error(child_node)

        node.leave(self.t)
        self.t += 1

        return node


def _compute_relative_measures(
    block: BlockVo, browser_context: BrowserContext, include_raw=False
) -> dict[str, float]:
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

    if include_raw:
        out |= {
            "center_x_raw": cx,
            "center_y_raw": cy,
            "top_left_x_raw": block.x,
            "top_left_y_raw": block.y,
            "bottom_right_x_raw": block.x + block.width,
            "bottom_right_y_raw": block.y + block.height,
            "width_raw": block.width,
            "height_raw": block.height,
        }

    out["is_visible"] = (0 <= block.x <= browser_context.window_size["width"]) and (
        0 <= block.x <= browser_context.window_size["height"]
    )

    return out


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
    
    # content heuristics
    

    return sum([x * w for x, w in zip(scores, weights)]) / sum(weights)


class PopulationLayout:
    center_x = 0
    center_y = 0

    def __init__(self, **kwargs) -> None:
        for key, val in kwargs.items():
            self.__setattr__(key, val)

    @staticmethod
    def from_block_list(
        block_list: List[BlockVo], browser_context
    ) -> "PopulationLayout":
        """Generates a population layout from a list of blocks and a browser context

        Args:
            block_list (List[BlockVo]): the list of blocks
            browser_context (_type_): the browser context

        Returns:
            PopulationLayout: the population layout
        """

        comp = [_compute_relative_measures(x, browser_context) for x in block_list]

        comp_x, comp_y, comp_area = zip(
            *[
                (x["center_x"] * x["area"], x["center_y"] * x["area"], x["area"])
                for x in comp
            ]
        )

        if sum(comp_area) == 0:
            return (0.5, 0.5)

        return PopulationLayout(
            center_x=sum(comp_x) / sum(comp_area), center_y=sum(comp_y) / sum(comp_area)
        )


def pre_score_blocks(
    blocks: List[BlockVo],
    browser_context: BrowserContext,
    scoring_method: Callable[[BlockVo, PopulationLayout, BrowserContext], float] = None,
    threshold: float = 0,
) -> Tuple[List[BlockVo], List[float]]:
    """scores the blocks with a heuristic

    Args:
        blocks (List[BlockVo]): the list of blocks to score
        browser_context (BrowserContext): the browser invariants
        scoring_method (function, optional): the heuristic used to give scores: it should take as arguments a block, PopulationLayout, and BrowserContext. Defaults to saliency_score.
        threshold (float, optional): the minimum score allowed to keep the block. Defaults to 0.

    Returns:
        Tuple[List[BlockVo], List[float]]: the list of blocks taken, the list of corresponding scores
    """

    pop_layout = PopulationLayout.from_block_list(blocks, browser_context)

    blocks_scores = [
        (
            b,
            scoring_method(
                b,
                pop_layout,
                browser_context,
            ),
        )
        for b in blocks
    ]

    blocks_to_add = [(b, s) for b, s in blocks_scores if s > threshold]

    if len(blocks_to_add) == 0:
        return [], []

    return zip(*blocks_to_add)


def draw_site_with_blocks(
    ax: Axes,
    blocks: List[BlockVo],
    labels: dict[str, int],
    site_screenshot: Image.Image,
    browser_context: BrowserContext,
):
    """Draws a representation of the site annotated with blocks to label.

    Args:
        ax (Axes): Pyplot Axes figure used
        blocks (List[BlockVo])
        scores (List[float])
        site_screenshot (Image.Image)
        browser_context (BrowserContext)
        block_debug (dict): A dictionary to specify the info to print regarding the blocks

        The allowed keys for `block_debug` and their meaning:
        - 'info' (set): contains the set of block attributes to print
            - 'score': print the score from the scoring method
            - 'position': print the position on screen and dimensions of the block
            - 'n_boxs': print the number of DomNode element in the block
        - 'box' (set):
            - 'attributes': tag attributes
            - 'position': the position on the screen
            - 'is_visible': boolean indicating if element considered visible
            - 'head': the first 20 characters of the text content
            - 'text' : the full length of the text
    """

    ax.clear()
    ax.imshow(site_screenshot)

    VERTICAL_RECT_OFFSET = 50

    for i, _b in enumerate(blocks):
        rect = patches.Rectangle(
            (
                _b.x / browser_context.window_size["width"] * site_screenshot.width,
                (_b.y + VERTICAL_RECT_OFFSET)
                / browser_context.window_size["height"]
                * site_screenshot.height,
            ),
            _b.width / browser_context.window_size["width"] * site_screenshot.width,
            _b.height / browser_context.window_size["width"] * site_screenshot.width,
            linewidth=1,
            edgecolor="r" if labels[_b.id] == 1 else "b",
            facecolor="none",
        )

        ax.add_patch(rect)
        ax.annotate(
            f"Block {_b.id}",
            (
                _b.x / browser_context.window_size["width"] * site_screenshot.width,
                _b.y / browser_context.window_size["height"] * site_screenshot.height,
            ),
            bbox={"color": "w", "alpha": 0.7},
        )


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io

    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


def save_screenshot_with_blocks(
    suffix: str,
    blocks: List[BlockVo],
    labels: dict[str, int],
    webdriver,
    manager_params,
    visit_id,
):
    img_str = webdriver.get_screenshot_as_base64()
    site_screenshot = Image.open(BytesIO(base64.b64decode(img_str)))

    fig, ax = plt.subplots(figsize=(14, 10))

    # process_website(blocks, scores, web_screenshot, browser_context, ax)

    draw_site_with_blocks(
        ax,
        blocks,
        labels,
        site_screenshot,
        BrowserContext(window_size=webdriver.get_window_size()),
    )

    # transform the figure to PIL image without saving to disk
    img = fig2img(fig)

    if suffix != "":
        suffix = "-" + suffix

    urlhash = md5(webdriver.current_url.encode("utf-8")).hexdigest()

    outname = os.path.join(
        manager_params.screenshot_path,
        "%i-%s%s.png" % (visit_id, urlhash, suffix),
    )

    # save the image to disk
    img.save(outname)
