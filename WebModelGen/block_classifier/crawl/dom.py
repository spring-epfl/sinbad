from __future__ import annotations
import base64
from io import BytesIO
import json
from pathlib import Path
from typing import List

from selenium.webdriver import Firefox
from selenium.webdriver.common.by import By

from .types import BlockVo, SegmentationGenerator, BrowserContext, DomNode
from vips.vips import Vips

from PIL import Image


__dir__ = Path(__file__).parent


class DomEncoder:
    """Based from PythonVispy implementation. Encodes the dom with visual attributes"""

    def __init__(self):
        self.nodeList = []
        self.count = 0
        self.html = None
        self.cssBoxList = dict()
        self.count3 = 0
        self.t = 0

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
                        pass

        node.leave(self.t)
        self.t += 1

        return node


def get_dump_js_script():
    with open(str(__dir__.joinpath("dom.js")), "r", encoding="utf-8") as f:
        js_script = f.read()

    js_script += (
        '\nreturn JSON.stringify(toJSON(document.getElementsByTagName("BODY")[0]));'
    )

    return js_script


def extract_blocks(
    webdriver: Firefox, SegGen: SegmentationGenerator = Vips, n_rounds: int = 5
) -> List[BlockVo]:

    with open(str(__dir__.joinpath("dom.js")), "r", encoding="utf-8") as f:
        js_script = f.read()

    js_script += (
        '\nreturn JSON.stringify(toJSON(document.getElementsByTagName("BODY")[0]));'
    )
    # finally run the javascript, and wait for it to finish and call the someCallback function.

    img_str = webdriver.get_screenshot_as_base64()
    js_resp = webdriver.execute_script(js_script)

    # initialize the DOM encoder
    encoder = DomEncoder()

    # encode the dom dump into DomNode tree
    encoder.to_dom(js_resp)

    # saliency edits
    vips = SegGen(
        encoder.nodeList, BrowserContext(window_size=webdriver.get_window_size())
    )

    vips.setRound(n_rounds)
    block_list = vips.service()

    site_screenshot = Image.open(BytesIO(base64.b64decode(img_str)))

    return block_list, site_screenshot
