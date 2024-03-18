import argparse
import json
import os
import sys
import traceback
from multiprocessing import Event, Process, Queue
from pathlib import Path
from types import FunctionType
from typing import List, Tuple
from tabulate import tabulate

import pandas as pd
from matplotlib import patches
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from PIL import Image

from .crawl.saliency import (
    saliency_score,
)
from .crawl.types import (
    BrowserContext,
    SegmentationGenerator,
    PopulationLayout,
    BlockVo,
)
from vips.vips import Vips
from .crawl.crawler import (
    load_website,
    setup_webdriver,
    load_websites,
)
from storage_dump.json_storage import JSONStorageController


AVAILABLE_SCORING_METHODS = {"saliency_score": saliency_score}
AVAILABLE_SEGMENTATION_GENERATORS = {"vips": Vips}


plt.ion()


def pre_score_blocks(
    blocks: List[BlockVo],
    browser_context: BrowserContext,
    scoring_method: FunctionType = saliency_score,
    threshold: float = 0,
) -> Tuple[List[BlockVo], List[float]]:
    """scores the blocks with a heuristic

    Args:
        blocks (List[BlockVo]): the list of blocks to score
        browser_context (BrowserContext): the browser invariants
        scoring_method (function, optional): the heuristic used to give scores: it should take as arguments a block, PopulationLayout, and BrowserContext. Defaults to saliency_score.
        threshold (float, optional): the minimum score allowed to keep the block. Defaults to 0.

    Returns:
        Tuple[List[BlockVo], List[float]]: the list of blocks taken, the list of corresponding scores
    """

    blocks_scores = [
        (
            b,
            scoring_method(
                b,
                PopulationLayout.from_block_list(blocks, browser_context),
                browser_context,
            ),
        )
        for b in blocks
    ]

    blocks_to_add = [(b, s) for b, s in blocks_scores if s > threshold]

    if len(blocks_to_add) == 0:
        return [], []

    return zip(*blocks_to_add)


def draw_site_with_blocks(
    ax: Axes,
    blocks: List[BlockVo],
    scores: List[float],
    site_screenshot: Image.Image,
    browser_context: BrowserContext,
    block_debug: dict = {
        "info": {"score", "position", "n_boxs"},
        "box": {"position", "attributes", "head"},
    },
):
    """Draws a representation of the site annotated with blocks to label.

    Args:
        ax (Axes): Pyplot Axes figure used
        blocks (List[BlockVo])
        scores (List[float])
        site_screenshot (Image.Image)
        browser_context (BrowserContext)
        block_debug (dict): A dictionary to specify the info to print regarding the blocks

        The allowed keys for `block_debug` and their meaning:
        - 'info' (set): contains the set of block attributes to print
            - 'score': print the score from the scoring method
            - 'position': print the position on screen and dimensions of the block
            - 'n_boxs': print the number of DomNode element in the block
        - 'box' (set):
            - 'attributes': tag attributes
            - 'position': the position on the screen
            - 'is_visible': boolean indicating if element considered visible
            - 'head': the first 20 characters of the text content
            - 'text' : the full length of the text
    """

    ax.clear()
    ax.imshow(site_screenshot)

    VERTICAL_RECT_OFFSET = 50

    for i, _b in enumerate(blocks):

        rect = patches.Rectangle(
            (
                _b.x / browser_context.window_size["width"] * site_screenshot.width,
                (_b.y + VERTICAL_RECT_OFFSET)
                / browser_context.window_size["height"]
                * site_screenshot.height,
            ),
            _b.width / browser_context.window_size["width"] * site_screenshot.width,
            _b.height / browser_context.window_size["width"] * site_screenshot.width,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )

        ax.add_patch(rect)
        ax.annotate(
            f"Block {_b.id}",
            (
                _b.x / browser_context.window_size["width"] * site_screenshot.width,
                _b.y / browser_context.window_size["height"] * site_screenshot.height,
            ),
            bbox={"color": "w", "alpha": 0.7},
        )

        if block_debug:
            print(
                f"\n--- Block {_b.id} -------------------------------------------------"
            )

            if "info" in block_debug:
                b_info = {
                    "score": scores[i],
                    "position": f"x={_b.x}, y={_b.y}, width={_b.width}, height={_b.height}",
                    "n_boxs": len(_b.boxs),
                }
                print(
                    tabulate(
                        [[b_info[k] for k in block_debug["info"]]],
                        headers=block_debug["info"],
                    )
                )

            if "box" in block_debug:

                print("\nBoxs:")

                for _x in _b.boxs:

                    print(f"\n- <{_x.nodeName}/>")

                    x_info = {
                        "attributes": _x.attributes,
                        "position": _x.visual_cues["bounds"],
                        "is_visible": _x.visual_cues["is_visible"],
                        "text": _x.visual_cues["text"]
                        if "text" in _x.visual_cues
                        else None,
                        "head": (
                            _x.visual_cues["text"][
                                : min(len(_x.visual_cues["text"]), 20)
                            ]
                            + "..."
                        )
                        if "text" in _x.visual_cues
                        else None,
                    }

                    for k, v in x_info.items():
                        if k in block_debug["box"]:
                            print(f"{k}: {v}")

    plt.show()


def process_website(
    blocks: List[BlockVo],
    scores: List[float],
    site_screenshot: Image.Image,
    browser_context: BrowserContext,
    ax: Axes,
) -> tuple[List[BlockVo], list[int]]:
    """Manual labeling process with user input

    Args:
        blocks (List[BlockVo])
        scores (List[float])
        site_screenshot (Image.Image)
        browser_context (BrowserContext)
        ax (Axes): pyplot axes figure used to show site

    Returns:
        tuple[List[BlockVo], list[int]]: [description]
    """

    # TODO: add modes to label both positive and negative and blocks to ignore

    draw_site_with_blocks(ax, blocks, scores, site_screenshot, browser_context)

    important_blocks = input(
        "label important blocks (ex: 101 102) or '-1' to skip site:"
    )
    important_blocks = important_blocks.split()

    # skip website
    if "-1" in important_blocks:
        return [], []

    labels = [0] * len(blocks)

    for i, b in enumerate(blocks):
        if b.id in important_blocks:
            labels[i] = 1

    return blocks, labels


def encode_blocks_to_dict(
    blocks: List[BlockVo],
    labels,
    website: str,
    browser_context: BrowserContext,
) -> List[dict]:
    """Create a dictionaries representations of Blocks to be saved into JSON

    Args:
        blocks (List[BlockVo])
        labels ([type])
        website (str)
        browser_context (BrowserContext)

    Returns:
        List[dict]
    """

    bds = []
    for block, label in zip(blocks, labels):
        bd = {
            k: block.__getattribute__(k)
            for k in {"id", "x", "y", "width", "height", "isVisualBlock"}
        }
        bd["label"] = label
        bd["boxs"] = [
            {
                k: box.__getattribute__(k)
                for k in {
                    "nodeType",
                    "tagName",
                    "nodeName",
                    "nodeValue",
                    "visual_cues",
                    "attributes",
                }
                if hasattr(box, k)
            }
            for box in block.boxs
        ]

        bd["url"] = website
        bd["context"] = browser_context.__dict__

        bds.append(bd)

    return bds


def get_processed_websites(processed_dir: str) -> set[str]:
    """Gets the websites processed in the previous run

    Args:
        processed_dir (str): directory of the previous run output

    Returns:
        set[str]: set of processed website urls
    """
    processed = set()

    if os.path.exists(processed_dir + "/data.json"):
        with open(processed_dir + "/data.json", "r") as f:
            data = json.load(f)
            processed = {d["url"] for d in data}

    return processed


def main(
    input_fp="websites.txt",
    output_dir="dataset-25.07.2022",
    overwrite=False,
    segmentor: SegmentationGenerator = Vips,
    scoring_method=saliency_score,
    scoring_threshold=0,
):
    print("[main] Loading generator")
    with open(input_fp, mode="r") as f:
        websites = f.readlines()

    websites = set(websites)

    processed = set()
    if not overwrite:
        processed = get_processed_websites(output_dir)
    else:
        os.remove(output_dir)

    websites = websites.difference(processed)

    fig, ax = plt.subplots(figsize=(14, 10))

    # Queues and events
    ready_queue = Queue()
    should_terminate = Event()

    # parallel block extraction process
    background_process = Process(
        target=load_websites, args=(ready_queue, should_terminate, websites, segmentor)
    )
    background_process.start()

    done = 0

    print("[main] Loading generator")

    # main labeling loop

    with JSONStorageController(
        Path("dataset-25.07.2022").resolve(),
        [
            "data.json",
        ],
        json_mode="list",
        is_async=False,
    ) as storage:
        while done < len(websites):

            print(f"[main] Waiting for website [{done}/{len(websites)}]")

            if done == 0:
                print("[main] Setting up browser...")

            while ready_queue.empty():
                pass

            website, out = ready_queue.get()

            if website == "ERROR":
                break

            if website != "404":
                (blocks, web_screenshot, browser_context) = out
                blocks, scores = pre_score_blocks(
                    blocks, browser_context, scoring_method, scoring_threshold
                )

                print(f"[main] Presenting ", website)
                # stuff with user input

                try:
                    blocks, labels = process_website(
                        blocks,
                        scores,
                        web_screenshot,
                        browser_context,
                        ax,
                    )

                    storage.save(
                        "data.json",
                        encode_blocks_to_dict(blocks, labels, website, browser_context),
                    )

                except EOFError:
                    plt.close()
                    should_terminate.set()
                    return

                except Exception:
                    plt.close()
                    traceback.print_exc()
                    should_terminate.set()
                    return

            else:
                print("[main] Page not found")

            done += 1

    # if error
    if done != len(websites):
        print("[main] error did not process all websites")

    should_terminate.set()


def debug_page(url, scoring_method=saliency_score, scoring_threshold=0):
    driver = setup_webdriver()
    (blocks, web_screenshot, browser_context) = load_website(
        url,
        driver,
    )

    blocks, scores = pre_score_blocks(
        blocks, browser_context, scoring_method, scoring_threshold
    )

    fig, ax = plt.subplots(figsize=(14, 10))

    process_website(blocks, scores, web_screenshot, browser_context, ax)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog=sys.argv[0], description="Run the labelling tool..."
    )

    parser.add_argument(
        "--i",
        type=str,
        help="path to a file containing the list of websites.",
        default="websites.txt",
    )
    parser.add_argument(
        "--o",
        type=str,
        help="the output result of the labeled dataset",
        default="dataset-25.07.2022",
    )

    parser.add_argument("--f", type=bool, default=False)
    parser.add_argument(
        "--segment",
        type=str,
        default="vips",
        help="The segmentation generator. you can create your own but it must conform to the SegmentationGenerator base class",
        choices=list(AVAILABLE_SEGMENTATION_GENERATORS.keys()),
    )
    parser.add_argument(
        "--filter",
        type=str,
        default="saliency_score",
        help="The manual pre-scoring method used to filter blocks that should appear for labeling",
        choices=list(AVAILABLE_SCORING_METHODS.keys()),
    )
    parser.add_argument(
        "--min",
        type=float,
        help="the minimum score for a block to appear in the list of blocks to be labelled.",
        default=0,
    )

    ns = parser.parse_args(sys.argv[1:])
    main(
        ns.i,
        ns.o,
        ns.f,
        AVAILABLE_SEGMENTATION_GENERATORS[ns.segment],
        AVAILABLE_SCORING_METHODS[ns.filter],
        ns.min,
    )
    exit()
