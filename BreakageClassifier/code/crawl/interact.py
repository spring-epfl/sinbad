from __future__ import annotations

import logging
import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime
from random import choices
from time import sleep
from typing import List, Tuple

import joblib
from block_classifier.crawl.types import BlockVo, BrowserContext, DomNode
from openwpm.commands.types import BaseCommand
from openwpm.config import BrowserParams, ManagerParams
from openwpm.socket_interface import ClientSocket
from selenium.webdriver import Firefox
from selenium.webdriver.common.action_chains import ActionChains
from openwpm.commands.browser_commands import close_other_windows, GetCommand
from openwpm.commands.utils.webdriver_utils import wait_until_loaded
from selenium.common.exceptions import (
    NoSuchElementException,
    NoSuchWindowException,
    StaleElementReferenceException,
)

INT_TABLE_NAME = "interaction_annotations"


class BadInteractionException(Exception):
    """Exception raised if the interaction is not valid"""

    pass


def create_interaction_table(manager_params: ManagerParams):
    """create the sqlite table for the DOM dump."""
    path = manager_params.data_directory / "crawl-data.sqlite"
    db = sqlite3.connect(path)
    cur = db.cursor()

    cur.execute(
        f"""CREATE TABLE IF NOT EXISTS {INT_TABLE_NAME} (
            browser_id INTEGER,
            visit_id INTEGER,
            node_id INTEGER,
            block_id INTEGER,
            type TEXT,
            timestamp float,
            page_change BOOLEAN
            );"""
    )
    cur.close()
    db.close()


def _save_to_db(interactions, browser_id, visit_id, manager_params: ManagerParams):
    # writing the list of nodes to database

    sock = ClientSocket(serialization="dill")
    assert manager_params.storage_controller_address is not None
    sock.connect(*manager_params.storage_controller_address)

    for interaction in interactions:
        sock.send(
            (
                INT_TABLE_NAME,
                {
                    "browser_id": browser_id,
                    "visit_id": visit_id,
                    "node_id": interaction["node_id"],
                    "block_id": interaction["block_id"],
                    "type": interaction["type"],
                    "timestamp": interaction["timestamp"],
                    "page_change": interaction["page_change"],
                },
            )
        )
    sock.close()


class Action(ABC):
    __slots__ = ["type"]

    @staticmethod
    def _get_node_center(node: DomNode):
        return (
            node.visual_cues["bounds"]["x"] + node.visual_cues["bounds"]["width"] / 2,
            node.visual_cues["bounds"]["y"] + node.visual_cues["bounds"]["height"] / 2,
        )

    @staticmethod
    def _get_selenium_node(webdriver: Firefox, node: DomNode):
        # get center coordinates
        x, y = Action._get_node_center(node)

        candidates = webdriver.execute_script(
            f"return document.elementsFromPoint({x}, {y})"
        )

        for candidate in candidates:
            try:
                if candidate.tag_name == node.nodeName:
                    return candidate
            except StaleElementReferenceException:
                pass

        # if not found don't click on anything
        return None

    @abstractmethod
    def evaluate(
        self, webdriver: Firefox, chain: ActionChains, node: DomNode, block: BlockVo
    ) -> None:
        raise NotImplementedError()


class ClickAction(Action):
    type = "click"

    def evaluate(
        self, webdriver: Firefox, chain: ActionChains, node: DomNode, block: BlockVo
    ) -> None:
        clickable_element = Action._get_selenium_node(webdriver, node)
        if clickable_element:
            chain.click(on_element=clickable_element)


class TypeAction(Action):
    type = "type"

    def evaluate(
        self, webdriver: Firefox, chain: ActionChains, node: DomNode, block: BlockVo
    ) -> None:
        writable_element = Action._get_selenium_node(webdriver, node)
        if writable_element:
            chain.click(writable_element)
            chain.send_keys(["t", "e", "s", "t", "i", "n", "g"])


class ScrollAction(Action):
    type = "scroll"

    def evaluate(
        self, webdriver: Firefox, chain: ActionChains, node: DomNode, block: BlockVo
    ) -> None:
        # chain.scroll_by_amount(webdriver.get_window_size()["height"] / 2)
        # TODO: Figure out how to deal with the version of selenium issue.
        raise NotImplementedError()


def _get_valid_interactions(tag, attributes, height):
    interactions = []

    if (
        tag in ["button", "iframe", "video", "img"]
        or tag == "input"
        and attributes["type"] in ["submit", "radio", "checkbox"]
    ):
        interactions.append(ClickAction())

    if (
        tag in ["textarea"]
        or tag == "input"
        and attributes["type"] in ["text", "password", "email", "number"]
    ):
        interactions.append(TypeAction())

    # COMMENTED OUT BECAUSE IT IS NOT IMPLEMENTED
    # |
    # V
    # if tag in ["section", "ul", "table", "div", "iframe"] and height > 0.3:
    #     interactions.append(ScrollAction())

    return interactions


def _rank_candidates(blocks: list[BlockVo], browser_context: BrowserContext):
    candidates_blocks_actions = []

    for block in blocks:
        for node in block.boxs:
            interactions = _get_valid_interactions(
                node.nodeName,
                {x["key"]: x["value"] for x in node.attributes},
                block.height / browser_context.window_size["height"],
            )

            if node.nodeName == "iframe":
                candidates_blocks_actions.append((node, block, ClickAction(), 4))

                for i in interactions:
                    if i.type != "click":
                        candidates_blocks_actions.append((node, block, i, 1))

            elif node.nodeName == "button":
                candidates_blocks_actions.append((node, block, ClickAction(), 3))

                for i in interactions:
                    if i.type != "click":
                        candidates_blocks_actions.append((node, block, i, 0))

            elif any(i.type == "click" for i in interactions):
                candidates_blocks_actions.append((node, block, ClickAction(), 2))

                for i in interactions:
                    if i.type != "click":
                        candidates_blocks_actions.append((node, block, i, 1))

            elif any(i.type == "type" for i in interactions):
                candidates_blocks_actions.append((node, block, TypeAction(), 1))
                for i in interactions:
                    if i.type != "type":
                        candidates_blocks_actions.append((node, block, i, 0.5))
            else:
                for i in interactions:
                    candidates_blocks_actions.append((node, block, i, 1))

    out = []
    scores = []
    for *x, score in sorted(
        candidates_blocks_actions, key=lambda x: x[3], reverse=True
    ):
        out.append(x)
        scores.append(score)

    return out, scores


def _handle_page_change(
    visit_id,
    browser_id,
    webdriver: Firefox,
    browser_params: BrowserParams,
    manager_params: ManagerParams,
    extension_socket: ClientSocket,
    intended_url: str,
    current_url: str,
    intended_tab: str,
    current_tab: str,
    t_wait=3,
):
    if current_url != intended_url:
        if current_tab != intended_tab:
            try:
                webdriver.switch_to.window(intended_tab)
                close_other_windows(webdriver)
            except NoSuchWindowException:
                raise BadInteractionException("Window closed")
        else:
            get_command = GetCommand(intended_url, t_wait)
            get_command.set_visit_browser_id(visit_id, browser_id)
            get_command.execute(
                webdriver, browser_params, manager_params, extension_socket
            )

        wait_until_loaded(webdriver, 300)
        return True

    else:
        # just refresh the page to undo the effects of the previous action
        webdriver.refresh()
        wait_until_loaded(webdriver, 300)

    return False


class InteractCommand(BaseCommand):
    def __init__(self, n_interactions=3, wait=3) -> None:
        self.logger = logging.getLogger("openwpm")
        self.n_interactions = n_interactions
        self.wait = wait

    @abstractmethod
    def execute(
        self,
        webdriver: Firefox,
        browser_params: BrowserParams,
        manager_params: ManagerParams,
        extension_socket: ClientSocket,
    ) -> None:
        raise NotImplementedError()

    def _interact(
        self,
        webdriver: Firefox,
        manager_params: ManagerParams,
        browser_params: BrowserParams,
        extension_socket: ClientSocket,
        candidates_blocks_actions: list[Tuple[DomNode, BlockVo, Action]],
        save_joblib: bool = False,
    ):
        # get the current url and tab handle
        current_url = webdriver.current_url
        current_tab = webdriver.current_window_handle

        self.logger.info("%i possible interactions", len(candidates_blocks_actions))

        # get the list of interaction report
        interactions = []

        # add the actions in the shared memory
        if save_joblib:
            joblib.dump(
                candidates_blocks_actions,
                manager_params.data_directory.joinpath("actions.joblib"),
            )

        _handle_page_change(
            self.visit_id,
            self.browser_id,
            webdriver,
            browser_params,
            manager_params,
            extension_socket,
            current_url,
            webdriver.current_url,
            current_tab,
            webdriver.current_window_handle,
        )

        for candidate, block, action in candidates_blocks_actions:
            # check if the previous action changed the url

            actions = ActionChains(webdriver)
            action.evaluate(webdriver, actions, candidate, block)
            actions.pause(self.wait)

            self.logger.info("%s @ {tag:%s}", type(action), candidate.nodeName)
            t = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            actions.perform()

            self.logger.debug("Before page change")
            # check if the last action changed the url
            changed = _handle_page_change(
                self.visit_id,
                self.browser_id,
                webdriver,
                browser_params,
                manager_params,
                extension_socket,
                current_url,
                webdriver.current_url,
                current_tab,
                webdriver.current_window_handle,
            )

            self.logger.debug("After page change")

            interactions.append(
                {
                    "timestamp": t,
                    "node_id": candidate.id,
                    "block_id": block.id,
                    "type": action.type,
                    "page_change": changed,
                }
            )

            if changed:
                self.logger.debug("Page change")

        _save_to_db(interactions, self.browser_id, self.visit_id, manager_params)


class SalientRandomInteractCommand(InteractCommand):
    # While this is not strictly necessary, we use the repr of a command for logging
    # So not having a proper repr will make your logs a lot less useful
    def __repr__(self) -> str:
        return "SalientRandomInteractCommand"

    # Have a look at openwpm.commands.types.BaseCommand.execute to see
    # an explanation of each parameter

    def execute(
        self,
        webdriver: Firefox,
        browser_params: BrowserParams,
        manager_params: ManagerParams,
        extension_socket: ClientSocket,
    ) -> None:
        candidates_blocks_actions, scores = _rank_candidates(
            webdriver.storage__salient_blocks,
            BrowserContext(window_size=webdriver.get_window_size()),
        )

        # apply a weighted randomized interaction approach

        if len(candidates_blocks_actions):
            candidates_blocks_actions = choices(
                candidates_blocks_actions,
                weights=scores,
                k=min(self.n_interactions, len(candidates_blocks_actions)),
            )

        self._interact(
            webdriver,
            manager_params,
            browser_params,
            extension_socket,
            candidates_blocks_actions,
            save_joblib=True,
        )


class SalientRepeatInteractCommand(InteractCommand):
    # While this is not strictly necessary, we use the repr of a command for logging
    # So not having a proper repr will make your logs a lot less useful
    def __repr__(self) -> str:
        return "SalientRepeatInteractCommand"

    # Have a look at openwpm.commands.types.BaseCommand.execute to see
    # an explanation of each parameter

    def execute(
        self,
        webdriver: Firefox,
        browser_params: BrowserParams,
        manager_params: ManagerParams,
        extension_socket: ClientSocket,
    ) -> None:
        # load the actions from previous command

        candidates_blocks_actions = joblib.load(
            manager_params.data_directory.joinpath("actions.joblib")
        )

        self._interact(
            webdriver,
            manager_params,
            browser_params,
            extension_socket,
            candidates_blocks_actions,
        )


class SelectorInteractCommand(BaseCommand):
    def __init__(self, css_selector) -> None:
        self.logger = logging.getLogger("openwpm")
        self.css_selector = css_selector

    # While this is not strictly necessary, we use the repr of a command for logging
    # So not having a proper repr will make your logs a lot less useful
    def __repr__(self) -> str:
        return "SelectorInteractCommand"

    # Have a look at openwpm.commands.types.BaseCommand.execute to see
    # an explanation of each parameter

    def execute(
        self,
        webdriver: Firefox,
        browser_params: BrowserParams,
        manager_params: ManagerParams,
        extension_socket: ClientSocket,
    ) -> None:
        # get the current url and tab handle
        current_url = webdriver.current_url
        current_tab = webdriver.current_window_handle

        if self.css_selector == "":
            self.logger.warning("CSS selector empty. Nothing to do.")
            return

        # testing for the filterlist

        try:
            first_button = webdriver.find_element_by_css_selector(self.css_selector)
        except NoSuchElementException:
            self.logger.error(f"No element found with selector {self.css_selector}")

        t = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

        first_button.click()
        first_button.click()
        first_button.click()

        _handle_page_change(
            self.visit_id,
            self.browser_id,
            webdriver,
            browser_params,
            manager_params,
            extension_socket,
            current_url,
            webdriver.current_url,
            current_tab,
            webdriver.current_window_handle,
        )

        _save_to_db(
            [
                {
                    "timestamp": t,
                    "node_id": 1,
                    "block_id": 1,
                    "type": "click",
                },
            ],
            self.browser_id,
            self.visit_id,
            manager_params,
        )
