# some tests
import os
import sqlite3
from pathlib import Path

import pandas as pd
from openwpm.command_sequence import CommandSequence
from openwpm.commands.browser_commands import GetCommand
from openwpm.config import BrowserParams, ManagerParams
from openwpm.storage.leveldb import LevelDbProvider
from openwpm.storage.sql_provider import SQLiteStorageProvider
from openwpm.task_manager import TaskManager

from pathlib import Path
import os

base_path = Path(os.path.join(os.path.split(os.path.abspath(__file__))[0]))


from BreakageClassifier.code.crawl.ablockers.addon import AddonCheckAndUUIDCommand, AddonSetupCommand, FilterListLoadCommand
from conf import UBLOCK_XPI_PATH
from dom import create_dom_table
from interact import (
    create_interaction_table,
)
from interact import SelectorInteractCommand
from error import AddErrorHandlerCommand, create_errors_table

if __name__ == "__main__":

    # limit all scripts from executing
    with open(str(base_path.joinpath("error-filterlist.txt").resolve()), "w") as f:
        f.write("/$script\n")

    browser_params = [BrowserParams(display_mode="headless")]
    # browser_params = [BrowserParams()]

    # Update browser configuration (use this for per-browser settings)
    for browser_param in browser_params:
        # Record HTTP Requests and Responses
        browser_param.http_instrument = True
        # Record cookie changes
        browser_param.cookie_instrument = True
        # Record Navigations
        browser_param.navigation_instrument = True
        # Record JS Web API calls
        browser_param.js_instrument = True
        # Record the callstack of all WebRequests made
        browser_param.callstack_instrument = True
        # Record DNS resolution
        browser_param.dns_instrument = True
        # save the javascript files
        browser_param.save_content = "script"

        browser_param.js_instrument_settings = [
            "collection_fingerprinting",
            {
                "window": {
                    "propertiesToInstrument": [
                        "name",
                        "localStorage",
                        "sessionStorage",
                    ],
                    "nonExistingPropertiesToInstrument": [
                        "last_log_event",
                    ],
                }
            },
        ]

    manager_params = ManagerParams(
        num_browsers=1,
        data_directory=Path("./datadir-error"),
        log_path=Path("./datadir-error/openwpm.log"),
    )

    with TaskManager(
        manager_params,
        browser_params,
        SQLiteStorageProvider(Path(f"./datadir-error/crawl-data.sqlite").resolve()),
        LevelDbProvider(Path(f"./datadir-error/content.ldb").resolve()),
        # logger_kwargs={"log_level_console": logging.DEBUG},
    ) as manager:

        create_dom_table(manager_params)
        create_interaction_table(manager_params)
        create_errors_table(manager_params)

        TEST_URL = "https://www.calculator.net/"

        # load the ublock extension
        command_sequence = CommandSequence(
            TEST_URL,
            site_rank=0,
        )

        command_sequence.append_command(
            AddonSetupCommand(path=str(Path(UBLOCK_XPI_PATH).absolute()))
        )

        command_sequence.append_command(AddonCheckAndUUIDCommand())

        command_sequence.append_command(
            FilterListLoadCommand(path="error-filterlist.txt")
        )

        command_sequence.append_command(GetCommand(url=TEST_URL, sleep=20), timeout=200)

        # add the errorr instrumentation
        command_sequence.append_command(AddErrorHandlerCommand(), timeout=90)

        command_sequence.append_command(
            SelectorInteractCommand("span.scinm[onclick='r(1)']"), timeout=40
        )

        manager.execute_command_sequence(command_sequence)

    # remove the temporary file
    os.remove("error-filterlist.txt")

    # query the database
    with sqlite3.connect("datadir-error/crawl-data.sqlite") as con:

        errors = pd.read_sql_query(
            """
            SELECT value
            from javascript
            where symbol = 'window.last_log_event' and operation='set'
        """,
            con,
        )

    # checking errors

    errors = list(errors["value"])

    try:
        assert any("ReferenceError: r is not defined" in x for x in errors)
        print("Test Successfull")

    except AssertionError as e:
        print("Test unsuccessfull")
