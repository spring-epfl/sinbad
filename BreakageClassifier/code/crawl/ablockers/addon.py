""" This file aims to demonstrate how to write custom commands in OpenWPM

Steps to have a custom command run as part of a CommandSequence

1. Create a class that derives from BaseCommand
2. Implement the execute method
3. Append it to the CommandSequence
4. Execute the CommandSequence

"""
from http.server import SimpleHTTPRequestHandler
from io import TextIOWrapper
import json
import logging
from pathlib import Path
import os
import socketserver
import threading
from typing import Callable, Optional
from grpc import server

from openwpm.commands.types import BaseCommand
from openwpm.config import BrowserParams, ManagerParams
from openwpm.socket_interface import ClientSocket
from selenium.webdriver import Firefox
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

__dir__ = Path(os.path.dirname(os.path.abspath(__file__)))

FILTERLIST_CHUNK_MAX_LINES = 10000


class NoAddonException(Exception):
    """Exception raised if the Ublock Addon is not loaded"""


class AddonSetupCommand(BaseCommand):
    """loads an ad-blocker command"""

    def __init__(
        self,
        name: str,
        id: str,
        xpi_path: Path,
    ) -> None:
        self.logger = logging.getLogger("openwpm")
        self.name = name
        self.id = id
        self.addon_path = xpi_path

    # While this is not strictly necessary, we use the repr of a command for logging
    # So not having a proper repr will make your logs a lot less useful
    def __repr__(self) -> str:
        return "AddonSetupCommand"

    # Have a look at openwpm.commands.types.BaseCommand.execute to see
    # an explanation of each parameter
    def execute(
        self,
        webdriver: Firefox,
        browser_params: BrowserParams,
        manager_params: ManagerParams,
        extension_socket: ClientSocket,
    ) -> None:
        # TODO: Move size to own command
        webdriver.set_window_size(1920, 1680)

        webdriver.install_addon(str(self.addon_path.absolute()))

        try:
            url = get_addon_uuid(webdriver=webdriver, id=self.id)
            webdriver.extra = {"adblocker-uuid": url}
            self.logger.info(f"{self.name} adblocker installed @ %s", url)
        except NoAddonException:
            self.logger.error(f"{self.name} adblocker NOT installed")
        except Exception as e:
            self.logger.error(f"An error occured: {e}")

        self.logger.info("Removing default lists")


class GetAddonUUIDCommand(BaseCommand):
    """loads the ublock command"""

    def __init__(self, id: str) -> None:
        self.logger = logging.getLogger("openwpm")
        self.id = id

    def __repr__(self) -> str:
        return "GetAddonUUIDCommand"

    def execute(
        self,
        webdriver: Firefox,
        browser_params: BrowserParams,
        manager_params: ManagerParams,
        extension_socket: ClientSocket,
    ) -> None:
        if "extra" in webdriver.__dict__ and "adblocker-uuid" in webdriver.extra:
            return

        try:
            url = get_addon_uuid(webdriver=webdriver, id=self.id)
            webdriver.extra = {"adblocker-uuid": url}
            self.logger.info(f"adblocker installed @ %s", url)
        except NoAddonException:
            self.logger.error(f"adblocker NOT installed")
        except Exception as e:
            self.logger.error(f"An error occured: {e}")


def get_addon_uuid(webdriver: Firefox, id: str) -> str:
    """gets the addon uuid from the browser

    Args:
        webdriver (Firefox)

    Raises:
        NoAddonException

    Returns:
        str: the uuid of the addon. ex: "moz-extension://5df5ca6d-ddd4-4a0f-acf3-665a2dbf2f98"
    """

    webdriver.get("about:debugging#/runtime/this-firefox")

    # wait for the page to load
    WebDriverWait(webdriver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, ".debug-target-item__detail"))
    )

    url = webdriver.execute_script(
        f"""
        // find the element that starts with 'Extension ID'
        const el = [...document.querySelectorAll(".debug-target-item__detail")].find(el=>el.textContent.includes("{id}"));
        
        if (!el) {{
            throw new Error("No addon with id {id}");
        }}
        
        return el.querySelector("a").href;
        """
    )

    url = url.replace("/manifest.json", "")

    return url


FILTERLIST_PORT = 9000


class SilentHTTPRequestHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        # Do nothing, effectively silencing the server's console output
        pass


def start_filterlist_server(filterlists_dir: Path):
    stop_event = threading.Event()

    def run_server_thread(fp, stop_event):
        os.chdir(fp)

        with socketserver.TCPServer(
            ("localhost", FILTERLIST_PORT), SilentHTTPRequestHandler
        ) as httpd:
            while not stop_event.is_set():
                httpd.handle_request()

            # stop the server
            httpd.server_close()

        # stop the server if exit signal is received

    server_thread = threading.Thread(
        target=run_server_thread, args=(filterlists_dir, stop_event)
    )
    server_thread.daemon = True
    server_thread.start()
    return stop_event


def _add_filterlist_chunk(
    webdriver: Firefox,
    stream: Optional[TextIOWrapper],
    prev_len=0,
    var_name="filterlist",
):
    """Adds a chunk of the filterlist stream to the browser console as a variable.
    This prevents the browser from crashing when loading a large filterlist

    Returns
     - the number of characters added to the filterlist
     - True if there are more lines to add, False otherwise

    """
    script_init = f"""
    window.{var_name} = "";
    """

    if stream is None:
        script = script_init
        webdriver.execute_script(script)
        return 0, False

    is_first_chunk = prev_len == 0

    script_insert = f"""
        return (c => {{
        let prev_len = window.{var_name}.length;
        window.{var_name} += c + '\\n';
        return [prev_len, window.{var_name}.length - prev_len];
        }})(arguments[0]);
        """

    if is_first_chunk:
        script = script_init + script_insert
    else:
        script = script_insert

    chunk = ""
    lines = 0

    while lines < FILTERLIST_CHUNK_MAX_LINES:
        line = stream.readline()
        if not line:
            break

        if line.startswith("! ") or line.startswith("# "):
            continue

        chunk += line
        lines += 1

    if not chunk:
        return prev_len, False

    prev_len_js, added_len_js = webdriver.execute_script(script, chunk)

    # make sure the chunk is encoded the same way in python and js
    # chunk = json.loads(json.dumps(chunk))

    # if added_len_js != len(chunk) or prev_len_js != prev_len or chunk_js != chunk:
    #     diff_part = ""
    #     for i in range(len(chunk_js)):
    #         if chunk_js[i] != chunk[i]:
    #             diff_part = chunk_js[i:]
    #             break

    #     print("------------------------------")
    #     print("js: ", json.dumps(chunk_js))
    #     print("py: ", json.dumps(chunk))
    #     print("diff: ", json.dumps(diff_part))
    #     print("len_js: ", len(chunk_js))
    #     print("len_py: ", len(chunk))
    #     print("prev_len: ", prev_len)
    #     print("stream.tell(): ", stream.tell())
    #     print("------------------------------")

    #     raise FilterlistException(
    #         f"Failed to load filterlist: mismatched lengths prev_len_js={prev_len_js} added_len_js={added_len_js} len(chunk)={len(chunk)} prev_len={prev_len}"
    #     )

    return prev_len + len(chunk), lines == FILTERLIST_CHUNK_MAX_LINES


def load_filter_list(
    webdriver: Firefox,
    path: Path,
    loading_function: Callable[[Firefox, str, str], None],
):
    """loads a filterlist from a file path to the Ublock Addon configuration file

    Args:
        path (Path): path to the filterlist file
    """


    with open(path, "r", encoding="utf-8") as f:
        loading_function(webdriver, webdriver.extra["adblocker-uuid"], f)


class FilterlistException(Exception):
    pass


def empty_filter_list(
    webdriver: Firefox, loading_function: Callable[[Firefox, str, str], None]
):
    loading_function(webdriver, webdriver.extra["adblocker-uuid"], None)


class FilterListLoadCommand(BaseCommand):
    """loads the filterlist command"""

    def __init__(self, path, loading_function=None, server_function=None) -> None:
        self.logger = logging.getLogger("openwpm")
        self.path = path
        self.loading_function = loading_function
        self.server_function = server_function

    # While this is not strictly necessary, we use the repr of a command for logging
    # So not having a proper repr will make your logs a lot less useful
    def __repr__(self) -> str:
        return "FilterListLoadCommand"

    # Have a look at openwpm.commands.types.BaseCommand.execute to see
    # an explanation of each parameter
    def execute(
        self,
        webdriver: Firefox,
        browser_params: BrowserParams,
        manager_params: ManagerParams,
        extension_socket: ClientSocket,
    ) -> None:
        # if the path is None, we just reset the addon without loading any list
        if self.loading_function:
            if self.path:
                load_filter_list(webdriver, self.path, self.loading_function)
            else:
                empty_filter_list(webdriver, self.loading_function)
        elif self.server_function:
            if self.path:
                self.server_function(
                    webdriver, webdriver.extra["adblocker-uuid"], self.path
                )

        self.logger.info("Filter list loaded successfully: %s", self.path)


class WaitCommand(BaseCommand):
    def __init__(self, seconds: int) -> None:
        self.logger = logging.getLogger("openwpm")
        self.seconds = seconds

    def __repr__(self) -> str:
        return f"WaitCommand({self.seconds})"

    def execute(
        self,
        webdriver: Firefox,
        browser_params: BrowserParams,
        manager_params: ManagerParams,
        extension_socket: ClientSocket,
    ) -> None:
        # check if the page is loaded
        WebDriverWait(webdriver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "body"))
        )

        self.logger.info(f"Waiting {self.seconds} seconds")
        webdriver.execute_async_script(
            f"""
            var done = arguments[0];
            setTimeout(done, {self.seconds * 1000});
            """
        )
