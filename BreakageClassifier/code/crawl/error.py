from __future__ import annotations

import logging
import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime
from random import choices
from time import sleep

from openwpm.commands.types import BaseCommand
from openwpm.config import BrowserParams, ManagerParams
from openwpm.socket_interface import ClientSocket
from selenium.webdriver import Firefox

from pathlib import Path
import os

base_path = Path(os.path.join(os.path.split(os.path.abspath(__file__))[0]))


INT_TABLE_NAME = "js_errors"


def create_errors_table(manager_params: ManagerParams):
    """create the sqlite table for the DOM dump."""
    path = manager_params.data_directory / "crawl-data.sqlite"
    db = sqlite3.connect(path)
    cur = db.cursor()

    cur.execute(
        f"""CREATE TABLE IF NOT EXISTS {INT_TABLE_NAME} (
            browser_id INTEGER,
            visit_id INTEGER,
            message TEXT,
            stack TEXT,
            src TEXT,
            level TEXT,
            timestamp DATE
            );"""
    )
    cur.close()
    db.close()


def _save_to_db(errors, browser_id, visit_id, manager_params: ManagerParams):
    # writing the list of nodes to database

    sock = ClientSocket(serialization="dill")
    assert manager_params.storage_controller_address is not None
    sock.connect(*manager_params.storage_controller_address)

    for error in errors:
        sock.send(
            (
                INT_TABLE_NAME,
                {
                    "browser_id": browser_id,
                    "visit_id": visit_id,
                    "message": error["message"],
                    "stack": error["stack"],
                    "src": error["src"],
                    "level": error["level"],
                    "timestamp": error["timestamp"],
                },
            )
        )

    sock.close()


class AddErrorHandlerCommand(BaseCommand):
    def __init__(self) -> None:
        self.logger = logging.getLogger("openwpm")

    # While this is not strictly necessary, we use the repr of a command for logging
    # So not having a proper repr will make your logs a lot less useful
    def __repr__(self) -> str:
        return "AddErrorHandlerCommand"

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
            str(base_path.joinpath("error.js").resolve()), "r", encoding="utf-8"
        ) as f:
            js_script = f.read()

        def wrap_in_script(js_script):
            return f"""
        (function (){"{"}
            let script = document.createElement('script');
            script.innerHTML = `
            {js_script}
            `;

            document.body.appendChild(script);
        {"}"}
        )()
        """

        sleep(5)
        wrapped_in = wrap_in_script(js_script)

        webdriver.execute_script(wrapped_in)
        webdriver._error_handler_url = webdriver.current_url
        sleep(5)

        script_response = webdriver.execute_script("return window.logEvents;")

        # test if successfull
        assert isinstance(
            script_response, list
        ), f"Script not loaded correctly: Expected list, found {type(script_response)}. original script {wrapped_in}"

        self.logger.info("Loaded Error instrumentation script")

        try:
            self.logger.debug(
                "script exectued fully =%s",
                webdriver.execute_script("return window.breakage_error_script_loaded"),
            )
        except:
            pass

        try:
            self.logger.debug(
                "window.onerror=%s",
                webdriver.execute_script("return window.onerror"),
            )
        except:
            pass

        try:
            self.logger.debug(
                "window.last_log_event=%s",
                webdriver.execute_script("return window.last_log_event"),
            )
        except:
            pass

        try:
            self.logger.debug(
                "window.logEvents=%s",
                webdriver.execute_script("return window.logEvents"),
            )
        except:
            pass


class HarvestErrorsCommand(BaseCommand):
    def __init__(self) -> None:
        self.logger = logging.getLogger("openwpm")

    # While this is not strictly necessary, we use the repr of a command for logging
    # So not having a proper repr will make your logs a lot less useful
    def __repr__(self) -> str:
        return "HarvestErrorsCommand"

    def execute(
        self,
        webdriver: Firefox,
        browser_params: BrowserParams,
        manager_params: ManagerParams,
        extension_socket: ClientSocket,
    ) -> None:

        assert (
            "_error_handler_url" in webdriver.__dict__
        ), "HarvestErrorsCommand must run after a AddErrorHandlerCommand"

        if webdriver._error_handler_url != webdriver.current_url:
            self.logger.warn("Page redirect. Unable to log errors...")
            return

        errors = webdriver.execute_script("return window.logEvents")

        if isinstance(errors, list):
            self.logger.info("Logged %i errors", len(errors))

            _save_to_db(errors, self.browser_id, self.visit_id, manager_params)

        else:
            self.logger.warn(
                "window.logEvents is %s: error instrumentation script did not initialize event handlers correctly in %s.",
                type(errors),
                webdriver.current_url,
            )
