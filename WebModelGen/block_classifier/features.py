import argparse
import json
import sys
from math import exp
from pathlib import Path
from typing import List

import pandas as pd

from .crawl.types import BrowserContext, PopulationLayout, BlockVo
from .crawl.utils import _compute_relative_measures


class BoxStruct:
    def __init__(self, **args):
        self.__dict__.update(args)


class BlockStruct:
    def __init__(self, **args):

        self.id: str
        self.x: float
        self.y: float
        self.width: float
        self.height: float
        self.isVisualBlock: bool

        for key, val in args.items():
            if key != "boxs":
                self.__setattr__(key, val)
            else:
                boxs = [BoxStruct(**x) for x in val]
                self.__setattr__(key, boxs)


def get_positional_features(
    relative_pos: dict, pop_layout: PopulationLayout
) -> dict[str, float]:
    return {
        "center_x": relative_pos["center_x"],
        "center_y": relative_pos["center_y"],
        "width": relative_pos["width"],
        "height": relative_pos["height"],
        "pop_center_x": pop_layout.center_x,
        "pop_center_y": pop_layout.center_y,
        "centrality": exp(
            -10
            * (
                (relative_pos["center_x"] - 0.5) ** 2
                + (relative_pos["center_y"] - 0.5) ** 2
            )
        ),
    }


def get_content_features(boxs: List[BlockVo]) -> dict[str, float]:

    num_nodes = len(boxs)

    # functional elements count
    functional = {"a", "img", "button", "iframe", "video"}
    text = {"span", "div", "p", "h1", "h2", "h3", "h4", "h5", "h6"}
    layout = {"table", "tr", "td", "div", "section", "header"}

    num_functional = len([b for b in boxs if b.tagName in functional])
    num_text = len([b for b in boxs if b.tagName in text])
    num_layout = len([b for b in boxs if b.tagName in layout])

    text_length = sum(
        [len(b.visual_cues["text"]) for b in boxs if "text" in b.visual_cues]
    )

    return {
        "num_nodes": num_nodes,
        "num_functional": num_functional / num_nodes,
        "num_text": num_text / num_nodes,
        "num_layout": num_layout / num_nodes,
        "text_length": text_length,
    }


def get_visual_features(boxs: List[BlockVo]) -> dict[str, float]:

    box_features = []

    for box in boxs:
        f = {
            "fs": float(box.visual_cues["font-size"].split("px")[0]),
            "fw": float(box.visual_cues["font-weight"]),
        }

        color = (
            box.visual_cues["background-color"]
            .split("(")[1]
            .split(")")[0]
            .replace(" ", "")
            .split(",")
        )
        if len(color) == 4:
            color, opacity = [float(x) for x in color[:3]], float(color[3])
        else:
            color, opacity = [float(x) for x in color], 1

        max_comp = max(color)
        min_comp = min(color)

        # vibrancy
        if max_comp != 0:
            f["vb"] = (
                (max_comp + min_comp) * (max_comp - min_comp) / max_comp * opacity / 255
            )
        else:
            f["vb"] = 0

        box_features.append(f)

    return pd.DataFrame(box_features).mean(axis=0).to_dict()


def get_block_features(
    block: BlockVo, blocks: List[BlockVo], browser_context: BrowserContext
) -> dict[str, float]:

    pop_layout = PopulationLayout.from_block_list(blocks, browser_context)

    # block parameters
    relative_pos = _compute_relative_measures(block, browser_context)

    features = {}
    # positional features

    features.update(get_positional_features(relative_pos, pop_layout))

    # content features

    features.update(get_content_features(block.boxs))

    # visual features

    features.update(get_visual_features(block.boxs))

    # TODO: you can add more features here

    return features


def make_block_df(blocks: List[BlockStruct]) -> pd.DataFrame:
    """Extract the block features from the list of dictionary representation of blocks

    Args:
        blocks (List[BlockStruct])

    Returns:
        pd.DataFrame: the feature vector
    """

    grouped_by_url = {}

    for b in blocks:
        if b.url not in grouped_by_url:
            grouped_by_url[b.url] = []

        grouped_by_url[b.url].append(b)

    bf = [
        get_block_features(
            block, grouped_by_url[block.url], BrowserContext(**block.context)
        )
        for block in blocks
    ]

    for i, b in enumerate(bf):
        b["label"] = blocks[i].label
        b["id"] = blocks[i].id
        b["url"] = blocks[i].url

    return pd.DataFrame(bf)


def load_blocks_from_json(filepath: str):
    with open(filepath, "r") as f:
        blocks = json.load(f)

    return [BlockStruct(**b) for b in blocks]


def dump_features_to_csv(features: pd.DataFrame, filepath: Path):

    filepath.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(str(filepath.resolve()))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog=sys.argv[0], description="Run the feature extraction tool..."
    )

    parser.add_argument(
        "--i",
        type=str,
        help="path to a file containing the JSON website dataset",
        default="dataset-25.07.2022/data.json",
    )
    parser.add_argument(
        "--o",
        type=Path,
        help="the output feature path of the labeled dataset",
        default=Path("dataset-25.07.2022/features.csv"),
    )

    ns = parser.parse_args(sys.argv[1:])

    blocks = load_blocks_from_json(ns.i)
    features = make_block_df(blocks)
    dump_features_to_csv(features, ns.o)

    exit()
