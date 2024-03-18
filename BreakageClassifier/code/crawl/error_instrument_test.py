# some tests
from pathlib import Path
from socket import timeout
from BreakageClassifier.code.crawl.ablockers.addon import (
    AddonSetupCommand,
    FilterListLoadCommand,
)
from conf import UBLOCK_XPI_PATH
from interact import (
    SalientRandomInteractCommand,
    create_interaction_table,
)

from interact import (
    SelectorInteractCommand
)

from error import (
    AddErrorHandlerCommand,
    HarvestErrorsCommand,
    create_errors_table,
)

from dom import create_dom_table


if __name__ == "__main__":
    from openwpm.command_sequence import CommandSequence
    from openwpm.commands.browser_commands import GetCommand
    from openwpm.config import BrowserParams, ManagerParams
    from openwpm.storage.sql_provider import SQLiteStorageProvider
    from openwpm.storage.leveldb import LevelDbProvider
    from openwpm.task_manager import TaskManager

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
    ) as manager:

        create_dom_table(manager_params)
        create_interaction_table(manager_params)
        create_errors_table(manager_params)

        TEST_URL = "https://howtoinstall.co/en/libasound2"

        # load the ublock extension
        command_sequence = CommandSequence(
            TEST_URL,
            site_rank=0,
        )

        command_sequence.append_command(
            AddonSetupCommand(path=str(Path(UBLOCK_XPI_PATH).absolute()))
        )

        # command_sequence.append_command(AddonCheckAndUUIDCommand())

        command_sequence.append_command(
            FilterListLoadCommand(path="error-filterlist.txt")
        )

        command_sequence.append_command(GetCommand(url=TEST_URL, sleep=20), timeout=200)

        command_sequence.append_command(AddErrorHandlerCommand())

        command_sequence.append_command(SelectorInteractCommand(css_selector=""))

        # command_sequence.append_command(
        #     SalientDomDumpCommand(
        #         model_path="../../../WebModelGen/block_classifier/pretrained-models/model-0.joblib",
        #     )
        # )

        # command_sequence.append_command(SalientRandomInteractCommand(), timeout=120)

        manager.execute_command_sequence(command_sequence)
