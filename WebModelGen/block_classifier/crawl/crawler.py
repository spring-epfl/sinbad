import os
from multiprocessing import Event, Queue
from pathlib import Path
from time import sleep
from typing import List, Tuple

from PIL import Image
from selenium.common.exceptions import WebDriverException
from selenium.webdriver import Firefox, FirefoxOptions
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary

from vips.vips import Vips
from .dom import extract_blocks

from .types import BrowserContext, SegmentationGenerator, BlockVo


def setup_webdriver(width=1980, height=1620):

    options = FirefoxOptions()
    options.headless = True

    if os.name == "nt":
        binary = FirefoxBinary("C:\\Program Files\\Mozilla Firefox\\firefox.exe")
        driver = Firefox(options=options, firefox_binary=binary)
    else:
        filepath = (
            Path(os.path.dirname(os.path.realpath(__file__)))
            .joinpath("../../")
            .resolve()
        )

        binary = FirefoxBinary(str(filepath.joinpath("firefox-bin/firefox").resolve()))
        print(filepath.joinpath("drivers/geckodriver").resolve())
        driver = Firefox(
            options=options,
            firefox_binary=binary,
            executable_path=str(filepath.joinpath("drivers/geckodriver").resolve()),
        )

    driver.set_window_size(width, height)
    return driver


def get_wellformed_url(url: str):

    if "https:" not in url:
        url = "https:" + url
    return url


def load_website(
    url: str, driver: Firefox, segmentor: SegmentationGenerator = Vips, wait=10
) -> Tuple[List[BlockVo], Image.Image, BrowserContext]:

    driver.implicitly_wait(wait)
    driver.get(get_wellformed_url(url))

    sleep(wait)

    browser_context = BrowserContext(window_size=driver.get_window_size())

    blocks, site_screenshot = extract_blocks(driver, segmentor)

    return blocks, site_screenshot, browser_context


def load_websites(
    ready_queue: Queue,
    should_terminate: Event,
    websites: List[str],
    segmentor: SegmentationGenerator = Vips,
):
    """Background Process to crawl the websites asynchronously and send the results to the queue

    Args:
        ready_queue (Queue): the queue for website crawls ready to be processed
        should_terminate (Event): event signaling that the parent process is closing
        websites (List[str]): website url list to load
    """

    driver = setup_webdriver()

    def before_exit(*args):
        print("[background-crawl] terminated")
        driver.quit()
        ready_queue.close()

    try:
        for website in websites:

            # check if main process stopped
            if should_terminate.is_set():
                before_exit()
                return
            try:
                ready_queue.put(
                    (
                        website,
                        load_website(website, driver, segmentor),
                    ),
                    block=True,
                )

            except WebDriverException as e:
                ready_queue.put(("404", ()), block=True)

    except Exception as e:
        ready_queue.put(("ERROR", ()), block=True)

    before_exit()
