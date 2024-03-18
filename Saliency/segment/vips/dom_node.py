from __future__ import annotations


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
        self.parentNode = None

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
