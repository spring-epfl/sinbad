# some tests
from pathlib import Path
from interact import SalientRandomInteractCommand, create_interaction_table

from dom import SalientDomDumpCommand, create_dom_table


if __name__ == "__main__":
    from openwpm.command_sequence import CommandSequence
    from openwpm.commands.browser_commands import GetCommand
    from openwpm.config import BrowserParams, ManagerParams
    from openwpm.storage.sql_provider import SQLiteStorageProvider
    from openwpm.storage.leveldb import LevelDbProvider
    from openwpm.task_manager import TaskManager

    browser_params = [BrowserParams(display_mode="headless")]

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

    manager_params = ManagerParams(
        num_browsers=1,
        data_directory=Path("./datadir-attention"),
        log_path=Path("./datadir-attention/openwpm.log"),
    )

    with TaskManager(
        manager_params,
        browser_params,
        SQLiteStorageProvider(Path(f"./datadir-attention/crawl-data.sqlite").resolve()),
        LevelDbProvider(Path(f"./datadir-attention/content.ldb").resolve()),
    ) as manager:

        create_dom_table(manager_params)
        create_interaction_table(manager_params)

        TEST_URL = "https://www.microsoft.com/en-us/research/publication/vips-a-vision-based-page-segmentation-algorithm/"
        # load the ublock extension

        command_sequence = CommandSequence(
            TEST_URL,
            site_rank=0,
        )

        command_sequence.append_command(GetCommand(url=TEST_URL, sleep=1))

        command_sequence.append_command(
            SalientDomDumpCommand(
                model_path="../../../WebModelGen/block_classifier/pretrained-models/model-0.joblib",
            )
        )

        command_sequence.append_command(SalientRandomInteractCommand())

        manager.execute_command_sequence(command_sequence)
