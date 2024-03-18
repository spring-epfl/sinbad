from __future__ import annotations
import json
import logging
import sqlite3
import time
from typing import Optional
import joblib

from openwpm.commands.types import BaseCommand
from openwpm.config import BrowserParams, ManagerParams
from openwpm.socket_interface import ClientSocket
from selenium.webdriver import Firefox
from Saliency.segment.vips.dom_node import DomNode
from block_classifier.crawl.types import BrowserContext
from pathlib import Path
import os

from Saliency.classify import SaliencyClassifier
from Saliency.utils import DomEncoder, save_screenshot_with_blocks

base_path = Path(os.path.join(os.path.split(os.path.abspath(__file__))[0]))


from .conf import *

DOM_TABLE_NAME = "dom_nodes"


def create_dom_table(manager_params: ManagerParams):
    """create the sqlite table for the DOM dump."""
    path = manager_params.data_directory / "crawl-data.sqlite"
    db = sqlite3.connect(path)
    cur = db.cursor()

    cur.execute(
        f"""CREATE TABLE IF NOT EXISTS {DOM_TABLE_NAME} (
            browser_id INTEGER,
            visit_id INTEGER,
            id INTEGER, 
            nodeName TEXT,
            type INTEGER, 
            attributes TEXT,
            visual_cues TEXT,
            parent_id INTEGER,
            t_enter INTEGER,
            t_leave INTEGER,
            saliency FLOAT,
            block TEXT
            );"""
    )
    cur.close()
    db.close()


def _walk_node_tree(root_node: DomNode, scores={}):
    # unpacking the tree into a list of nodes with pointers to parent node
    nodes = []

    def __walk(node, parent_id=-1, parent_salient=False):
        salient = (
            parent_salient == 1.0
            or scores.get(node.id, {"score": None})["score"] == 1.0
        )

        nodes.append(
            {
                "id": node.id,
                "nodeName": node.nodeName,
                "type": node.nodeType,
                "attributes": node.attributes,
                "visual_cues": node.visual_cues,
                "parent_id": parent_id,
                "t_enter": node.t_enter,
                "t_leave": node.t_leave,
                "saliency": 1.0 if salient else 0.0,
                "block": scores.get(node.id, {"block": None})["block"],
            }
        )

        for _, child in enumerate(node.childNodes):
            __walk(child, node.id, salient)

    __walk(root_node)

    return nodes


def _save_to_db(nodes, browser_id, visit_id, manager_params: ManagerParams):
    # writing the list of nodes to database

    sock = ClientSocket(serialization="dill")
    assert manager_params.storage_controller_address is not None
    sock.connect(*manager_params.storage_controller_address)

    for node in nodes:
        sock.send(
            (
                DOM_TABLE_NAME,
                {
                    "browser_id": browser_id,
                    "visit_id": visit_id,
                    "id": int(node["id"]),
                    "nodeName": node["nodeName"],
                    "type": node["type"],
                    "attributes": json.dumps(node["attributes"]),
                    "visual_cues": json.dumps(node["visual_cues"]),
                    "parent_id": int(node["parent_id"]),
                    "t_enter": node["t_enter"],
                    "t_leave": node["t_leave"],
                    "saliency": node["saliency"],
                    "block": node["block"],
                },
            )
        )

    sock.close()


class DomDumpCommand(BaseCommand):
    """loads the ublock command"""

    def __init__(self) -> None:
        self.logger = logging.getLogger("openwpm")
        self.created_table = False

    # While this is not strictly necessary, we use the repr of a command for logging
    # So not having a proper repr will make your logs a lot less useful
    def __repr__(self) -> str:
        return "DomDumpCommand"

    # Have a look at openwpm.commands.types.BaseCommand.execute to see
    # an explanation of each parameter

    def execute(
        self,
        webdriver: Firefox,
        browser_params: BrowserParams,
        manager_params: ManagerParams,
        extension_socket: ClientSocket,
    ) -> None:
        with open(
            str(base_path.joinpath("dom.js").resolve()), "r", encoding="utf-8"
        ) as f:
            js_script = f.read()

        js_script += (
            '\nreturn JSON.stringify(toJSON(document.getElementsByTagName("BODY")[0]));'
        )
        # finally run the javascript, and wait for it to finish and call the someCallback function.
        js_resp = webdriver.execute_script(js_script)

        # initialize the DOM encoder
        encoder = DomEncoder(self.logger)

        # encode the dom dump into DomNode tree
        parent_node = encoder.to_dom(js_resp)

        # unpacking the tree into a list of nodes with pointers to parent node
        nodes = _walk_node_tree(parent_node)

        _save_to_db(nodes, self.browser_id, self.visit_id, manager_params)


class SalientDomDumpCommand(BaseCommand):
    def __init__(
        self, classifier: SaliencyClassifier, screenshot_suffix: Optional[str] = None
    ) -> None:
        self.logger = logging.getLogger("openwpm")
        self.created_table = False
        self.classifier = classifier
        self.screenshot_suffix = screenshot_suffix
        
        self.classifier.set_logger(self.logger)

    # While this is not strictly necessary, we use the repr of a command for logging
    # So not having a proper repr will make your logs a lot less useful
    def __repr__(self) -> str:
        return "SalientDomDumpCommand"

    # Have a look at openwpm.commands.types.BaseCommand.execute to see
    # an explanation of each parameter

    def execute(
        self,
        webdriver: Firefox,
        browser_params: BrowserParams,
        manager_params: ManagerParams,
        extension_socket: ClientSocket,
    ) -> None:
        with open(
            str(base_path.joinpath("dom.js").resolve()), "r", encoding="utf-8"
        ) as f:
            js_script = f.read()

        js_script += (
            '\nreturn JSON.stringify(toJSON(document.getElementsByTagName("BODY")[0]));'
        )
        
        t_start = time.time()
        
        # finally run the javascript, and wait for it to finish and call the someCallback function.
        js_resp = webdriver.execute_script(js_script)
        
        self.logger.debug("DOM dump received in %f seconds", time.time() - t_start)

        t_start = time.time()

        # classify the blocks with the saliency classifier
        block_scores, block_list, parent_node = self.classifier.predict_json(
            js_resp, BrowserContext(window_size=webdriver.get_window_size()),
        )
        
        self.logger.debug("Saliency classifier finished in %f seconds", time.time() - t_start)

        # remove all unlabeled blocks because they did not pass the pre-scoring heuristic
        block_list = [block for block in block_list if block.id in block_scores]

        t_start = time.time()

        # cast the score of the block to the nodes that are part of the block
        nodes_scores = {}

        for block in block_list:
            nodes_scores.update(
                {
                    n.id: {"score": block_scores[block.id], "block": block.id}
                    for n in block.boxs
                }
            )
            
        self.logger.debug("Block scores casted to nodes in %f seconds", time.time() - t_start)

        t_start = time.time()

        nodes = _walk_node_tree(parent_node, nodes_scores)
        
        self.logger.debug("DOM tree unpacked in %f seconds", time.time() - t_start)

        salient_blocks = [block for block in block_list if block_scores[block.id] == 1]

        webdriver.storage__salient_blocks = salient_blocks

        self.logger.info(
            "Found %i salient elements", len(webdriver.storage__salient_blocks)
        )

        _save_to_db(nodes, self.browser_id, self.visit_id, manager_params)

        if self.screenshot_suffix is not None:
            save_screenshot_with_blocks(
                self.screenshot_suffix,
                block_list,
                block_scores,
                webdriver,
                manager_params,
                self.visit_id,
            )
