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
from typing import Optional

from openwpm.commands.types import BaseCommand
from openwpm.config import BrowserParams, ManagerParams
from openwpm.socket_interface import ClientSocket
from selenium.webdriver import Firefox
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from . import addon

__dir__ = Path(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_XPI_PATH = __dir__ / "xpi/adguard_adblocker-4.2.209.xpi"
ID = "adguardadblocker@adguard.com"


class AddonSetupCommand(addon.AddonSetupCommand):
    def __init__(
        self,
        xpi_path: Optional[Path] = None,
    ) -> None:
        self.logger = logging.getLogger("openwpm")

        super().__init__(
            "adguard", "adguardadblocker@adguard.com", xpi_path or DEFAULT_XPI_PATH
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
    webdriver.get(f"{addon_uid}/pages/options.html#filters")

    webdriver.execute_script(
        """
        document.querySelectorAll(".checkbox__in").forEach(el=> {
            if (el.checked) {
                el.click();
            }
        })
    """
    )


def _load_filterlist_from_stream(
    webdriver: Firefox, addon_uid: str, filterlist_file: TextIOWrapper
):
    webdriver.get(f"{addon_uid}/pages/fullscreen-user-rules.html?theme=system")

    WebDriverWait(webdriver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, ".ace_content"))
    )

    more_to_write = True
    prev_len = 0

    while more_to_write:
        prev_len, more_to_write = addon._add_filterlist_chunk(
            webdriver, filterlist_file, prev_len
        )

    script = """
        var done = arguments[0];
        (async () => {
        try{
        await browser.runtime.sendMessage(
            {
                handlerName: 'app',
                type: 'saveUserRules',
                data: {
                    'value': window.filterlist
                }
            }
        );
        }catch(e){
            done(false);
            }
        
        done(true);
        
        })();
        
        """

    ## It is expensive to get the filterlist back from the extension
    # reply_js = """
    # var done = arguments[0];
    #     (async () => {
    # const reply = await browser.runtime.sendMessage({
    #         handlerName: 'app',
    #         type: 'getUserRules',
    #     });

    #     done(reply);
    #     })();
    # """

    webdriver.set_script_timeout(60)
    out = webdriver.execute_async_script(script)
    # reply = webdriver.execute_async_script(reply_js)
    # print("filterlist length", len(reply['content']))
    # if "content" not in out or out["content"].strip() != filterlist.strip():
    #     # get the similarity metric between the two strings
    #     raise addon.FilterlistException("Failed to load filterlist")

    if not out:
        raise addon.FilterlistException("Failed to load filterlist")


# def _load_filterlist_from_string(webdriver: Firefox, addon_uid: str, filterlist: str):
#     # if the filterlist is large, we need to send it in chunks

#     for start in range(0, len(filterlist), FILTERLIST_CHUNK_MAX_LINES):
#         end = min(start + FILTERLIST_CHUNK_MAX_LINES, len(filterlist))

#     json_str = json.dumps(
#         {
#             "handlerName": "app",
#             "type": "saveUserRules",
#             "data": {"value": filterlist},
#         }
#     )

#     script = (
#         """
#         var done = arguments[0];
#         (async () => {
#         await browser.runtime.sendMessage("""
#         + json_str
#         + """);
#         const reply = await browser.runtime.sendMessage({
#             handlerName: 'app',
#             type: 'getUserRules',
#         });

#         done(reply);
#         })();
#     """
#     )

#     webdriver.get(f"{addon_uid}/pages/fullscreen-user-rules.html?theme=system")

#     WebDriverWait(webdriver, 10).until(
#         EC.presence_of_element_located((By.CSS_SELECTOR, ".ace_content"))
#     )

#     out = webdriver.execute_async_script(script)

#     if "content" not in out or out["content"].strip() != filterlist.strip():
#         # get the similarity metric between the two strings

#         raise addon.FilterlistException("Failed to load filterlist")


def _filterlist_from_server_chunk_script(
    webdriver, filterlist_path: str, chunk_id: int
):
    issue_id = filterlist_path.parent.name
    filename = f"{filterlist_path.stem}_{chunk_id}.txt"

    if not os.path.exists(filterlist_path.parent / filename):
        return False

    url = f"http://localhost:{addon.FILTERLIST_PORT}/{issue_id}/{filename}"

    script = f"let done = arguments[0]; (async ()=>{{try{{await window.subscribe('{url}');}}catch(e){{done(false)}} done(true);}})();"

    webdriver.set_script_timeout(60)
    webdriver.execute_async_script(script)

    print(f"Loaded {url}")

    return True


def _set_filterlist_from_server(
    webdriver: Firefox, addon_id: str, filterlist_path: str
):
    """Loads the filterlists from the local HTTP server hosted at the adguard/data/filterlists directory

    Args:
        webdriver (Firefox)
        addon_id (str)
        filterlist_path (str): absolute path to the filterlist
    """

    # visit the page
    webdriver.get(f"{addon_id}/pages/options.html#filters?group=0")

    WebDriverWait(webdriver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, ".title"))
    )

    # rem srcipt
    rem_script = """
    // remove all previous filters
        document.querySelectorAll("a.filter__remove").forEach(el=> {
            el.click();
            
            document.querySelector("[role='dialog'] .button--red-bg").click();
        });
    """

    if not filterlist_path:
        webdriver.execute_script(rem_script)
        return

    # path relative to the filterlists directory

    # set the filterlist

    script = rem_script + (
        """
    var done = arguments[0];
        (async () => {
            
        window.subscribe = async (url) => {
        // validate the filterlist
        let filter_info = await browser.runtime.sendMessage(
            {
                handlerName: 'app',
                type: 'loadCustomFilterInfo',
                data: {
                    url: url
                    }
            }
        );
        
        if (! filter_info.filter) {
            throw new Error('Failed to load filterlist');
        }
        
        let filter = filter_info.filter;
        filter.trusted = true;
        filter.name = filter.customUrl;
        
        // add the filter
        
        await browser.runtime.sendMessage(
            {
                handlerName: 'app',
                type: 'subscribeToCustomFilter',
                data: {
                    filter: filter
                    }
            }
        );
        }
        
        // we need to enable the filter through the UI because we can't access the API for it
        let chkbx = document.querySelector('.checkbox__in[id="0"]');
        if (! chkbx.checked) {
            chkbx.click();
        }
        
        done(true);
        
        })();

    """
    )

    webdriver.set_script_timeout(60)
    webdriver.execute_async_script(script)

    chunk_id = 0
    while _filterlist_from_server_chunk_script(webdriver, filterlist_path, chunk_id):
        chunk_id += 1


class FilterListLoadCommand(addon.FilterListLoadCommand):
    """loads the filterlist command"""

    def __init__(self, path) -> None:
        super().__init__(path, loading_function=_load_filterlist_from_stream)
