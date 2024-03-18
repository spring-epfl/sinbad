from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List

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


from .utils import _compute_relative_measures


class PopulationLayout:

    center_x = 0
    center_y = 0

    def __init__(self, **kwargs) -> None:
        for key, val in kwargs.items():
            self.__setattr__(key, val)

    @staticmethod
    def from_block_list(block_list: List[BlockVo], browser_context) -> PopulationLayout:

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
    def service(self):
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
