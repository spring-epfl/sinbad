""" This file aims to demonstrate how to write custom commands in OpenWPM

Steps to have a custom command run as part of a CommandSequence

1. Create a class that derives from BaseCommand
2. Implement the execute method
3. Append it to the CommandSequence
4. Execute the CommandSequence

"""

from io import TextIOWrapper
import json
import logging
import sys
from pathlib import Path
import os
from typing import Callable, Optional

from openwpm.commands.types import BaseCommand
from openwpm.config import BrowserParams, ManagerParams
from openwpm.socket_interface import ClientSocket
from selenium.webdriver import Firefox
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from . import addon

__dir__ = Path(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_XPI_PATH = __dir__ / "xpi/ublock_origin-1.43.0.xpi"
ID = "uBlock0@raymondhill.net"


class AddonSetupCommand(addon.AddonSetupCommand):
    def __init__(
        self,
        xpi_path: Optional[Path] = None,
    ) -> None:
        self.logger = logging.getLogger("openwpm")

        super().__init__(
            "ublock", "uBlock0@raymondhill.net", xpi_path or DEFAULT_XPI_PATH
        )

    # Have a look at openwpm.commands.types.BaseCommand.execute to see
    # an explanation of each parameter
    def execute(
        self,
        webdriver: Firefox,
        browser_params: BrowserParams,
        manager_params: ManagerParams,
        extension_socket: ClientSocket,
    ) -> None:
        super().execute(webdriver, browser_params, manager_params, extension_socket)

        remove_default_lists(webdriver, webdriver.extra["adblocker-uuid"])


def remove_default_lists(webdriver: Firefox, addon_uid: str):
    webdriver.get(f"{addon_uid}/dashboard.html#3p-filters.html")

    webdriver.switch_to.frame("iframe")
    WebDriverWait(webdriver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "#buttonApply"))
    )

    out = webdriver.execute_script(
        """
        let i = 0;
        document.querySelectorAll("[type='checkbox']").forEach(el=> {
            if (i > 4 && el.checked) {
                el.click();
            }
            i++;
        })
        
        document.querySelector("#buttonApply").click();
        
        return i;
    """
    )

    webdriver.switch_to.parent_frame()


def _load_filterlist_from_stream(
    webdriver: Firefox, addon_uid: str, filterlist_file: Optional[TextIOWrapper] = None
):
    

    script = """
        var done = arguments[0];
        (async () => {
        
        const reply = await vAPI.messaging.send('dashboard', 
        {
            what: 'writeUserFilters',
            content: window.filterlist
        }
        );
        
        await vAPI.messaging.send('dashboard', {
            what: 'reloadAllFilters',
        });
        
        done(true);
        
        })();
    """

    webdriver.get("about:debugging#/runtime/this-firefox")
    webdriver.get(f"{addon_uid}/dashboard.html#1p-filters.html")

    WebDriverWait(webdriver, 10).until(
        EC.frame_to_be_available_and_switch_to_it((By.ID, "iframe"))
    )
    WebDriverWait(webdriver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "script[src='js/vapi.js']"))
    )
    
    more_to_write = True
    prev_len = 0

    while more_to_write:
        prev_len, more_to_write = addon._add_filterlist_chunk(
            webdriver, filterlist_file, prev_len
        )

    success = webdriver.execute_async_script(script)
    

    if success != True:
        raise addon.FilterlistException("Failed to load filterlist")

    webdriver.switch_to.parent_frame()


def _load_filterlist_from_string(webdriver: Firefox, addon_uid: str, filterlist: str):
    json_str = json.dumps({"what": "writeUserFilters", "content": filterlist})

    script = (
        """
        var done = arguments[0];
        (async () => {
        const reply = await vAPI.messaging.send('dashboard', """
        + json_str
        + """);
        vAPI.messaging.send('dashboard', {
            what: 'reloadAllFilters',
        });
        done(reply);
        })();
    """
    )
    webdriver.get("about:debugging#/runtime/this-firefox")
    webdriver.get(f"{addon_uid}/dashboard.html#1p-filters.html")

    WebDriverWait(webdriver, 10).until(
        EC.frame_to_be_available_and_switch_to_it((By.ID, "iframe"))
    )
    WebDriverWait(webdriver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "script[src='js/vapi.js']"))
    )

    out = webdriver.execute_async_script(script)

    if "content" not in out or out["content"].strip() != filterlist.strip():
        # get the similarity metric between the two strings

        raise addon.FilterlistException("Failed to load filterlist")

    webdriver.switch_to.parent_frame()


class FilterListLoadCommand(addon.FilterListLoadCommand):
    """loads the filterlist command"""

    def __init__(self, path) -> None:
        super().__init__(path, loading_function=_load_filterlist_from_stream)
