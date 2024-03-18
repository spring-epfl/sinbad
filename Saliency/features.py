import argparse
import json
import math
import sys
from math import exp
from pathlib import Path
from typing import List
from xml.etree.ElementInclude import include

import pandas as pd

from Saliency.utils import BrowserContext, PopulationLayout, BlockVo, _compute_relative_measures


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
    relative_pos: dict, pop_layout: PopulationLayout, include_raw=False
) -> dict[str, float]:
    return {
        "center_x": relative_pos["center_x"],
        "center_y": relative_pos["center_y"],
        "width": relative_pos["width"],
        "height": relative_pos["height"],
        "size": relative_pos["width"] * relative_pos["height"],
        "pop_center_x": pop_layout.center_x,
        "pop_center_y": pop_layout.center_y,
        "centrality": exp(
            -10
            * (
                (relative_pos["center_x"] - 0.5) ** 2
                + (relative_pos["center_y"] - 0.5) ** 2
            )
        ),
    } | {k: v for k, v in relative_pos.items() if include_raw and k.endswith("_raw")}


def _shannon_entropy(text: str) -> float:
    freq = {}

    for c in text:
        if c not in freq:
            freq[c] = 0

        freq[c] += 1

    return -sum([f / len(text) * math.log(f / len(text)) for f in freq.values()])


def attr_list_to_dict(attr_list: List[dict]) -> dict[str, str]:
    attr_dict = {}

    if attr_list is None:
        return attr_dict

    for attr in attr_list:
        key = attr["key"]
        attr_dict[key] = attr["value"]

    return attr_dict


def has_attr(box: BlockVo, attr: str) -> bool:
    attr_dict = attr_list_to_dict(box.attributes)
    return attr in attr_dict


def count_classes(box: BlockVo) -> int:
    attr_dict = attr_list_to_dict(box.attributes)
    if "class" in attr_dict:
        return len(attr_dict["class"].split(" "))
    else:
        return 0


def classes_entropy(box: BlockVo) -> float:
    attr_dict = attr_list_to_dict(box.attributes)
    if "class" in attr_dict:
        return _shannon_entropy(attr_dict["class"])
    else:
        return 0


def get_content_features(
    boxs: List[BlockVo],
    tag_groups,
    include_raw=False,
) -> dict[str, float]:
    num_nodes = len(boxs)

    # functional elements count
    # functional = {"a", "img", "button", "iframe", "video"}
    # text = {"span", "div", "p", "h1", "h2", "h3", "h4", "h5", "h6"}
    # layout = {"table", "tr", "td", "div", "section", "header"}

    tag_group_counts = {
        "num_" + k: len([b for b in boxs if b.tagName in v])
        for k, v in tag_groups.items()
    }

    total_tags = sum(tag_group_counts.values())

    if total_tags != 0:
        tag_group_counts = {k: v / total_tags for k, v in tag_group_counts.items()}

    # num_functional = len([b for b in boxs if b.tagName in functional])
    # num_text = len([b for b in boxs if b.tagName in text])
    # num_layout = len([b for b in boxs if b.tagName in layout])

    texts = " ".join([b.visual_cues["text"] for b in boxs if "text" in b.visual_cues])

    text_length = sum(
        [len(b.visual_cues["text"]) for b in boxs if "text" in b.visual_cues]
    )

    _classes_count = sum([count_classes(b) for b in boxs])
    _classes_entropy = sum([classes_entropy(b) for b in boxs])

    out = {
        "num_nodes": num_nodes,
        # "num_functional": num_functional / num_nodes,
        # "num_text": num_text / num_nodes,
        # "num_layout": num_layout / num_nodes,
        "text_length": text_length,
        "text_entropy": _shannon_entropy(texts),
        "has_id": any([has_attr(b, "id") for b in boxs]),
        "classes_count": _classes_count,
        "classes_ratio": _classes_count / num_nodes,
        "classes_entropy": _classes_entropy / num_nodes,
    } | tag_group_counts

    if include_raw:
        out |= {
            "tag_names": json.dumps([b.tagName for b in boxs]),
            "text": json.dumps(
                [b.visual_cues["text"] for b in boxs if "text" in b.visual_cues]
            ),
        }

    return out


def get_visual_features(boxs: List[BlockVo], include_raw=False) -> dict[str, float]:
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

    out = pd.DataFrame(box_features).mean(axis=0).to_dict()

    if include_raw:
        out |= {
            "font_sizes": json.dumps([f["fs"] for f in box_features]),
            "font_weights": json.dumps([f["fw"] for f in box_features]),
            "vibrancies": json.dumps([f["vb"] for f in box_features]),
            "background_colors": json.dumps(
                [
                    box.visual_cues["background-color"]
                    for box in boxs
                    if "background-color" in box.visual_cues
                ]
            ),
        }

    return out


def pop_features(df, tag_groups):
    num_nodes_total = df["num_nodes"].sum()

    out_features = {
        "num_nodes_pop": num_nodes_total,
    }

    tag_group_names = [f"num_{k}" for k in tag_groups.keys()]

    for tag_group_name in tag_group_names:
        out_features[tag_group_name + "_pop"] = (
            df[tag_group_name] * df["num_nodes"] / num_nodes_total
        ).sum()

    return pd.Series(out_features)


def update_features_with_population(features_df: pd.DataFrame, tag_groups):
    if "url" in features_df.columns:
        group_features = (
            features_df.groupby("url")
            .apply(lambda x: pop_features(x, tag_groups))
            .reset_index()
        )

        features_df = features_df.merge(group_features, on="url")

    else:
        group_features = pop_features(features_df, tag_groups)
        for k, v in group_features.items():
            features_df[k] = v

    for tag_group_name in tag_groups:
        features_df[tag_group_name + "_ratio"] = (
            features_df[f"num_{tag_group_name}"]
            * features_df[f"num_nodes"]
            / features_df[f"num_nodes_pop"]
        )

        features_df[tag_group_name + "_ratio2"] = (
            features_df[f"{tag_group_name}_ratio"]
            / features_df[f"num_{tag_group_name}_pop"]
        )

    # if NaN, then the population is 0, so the ratio is 0
    features_df = features_df.fillna(0)

    return features_df


def get_block_features(
    block: BlockVo,
    blocks: List[BlockVo],
    browser_context: BrowserContext,
    tag_groups,
    include_raw=False,
) -> dict[str, float]:
    pop_layout = PopulationLayout.from_block_list(blocks, browser_context)

    # block parameters
    relative_pos = _compute_relative_measures(block, browser_context, include_raw)

    features = {}
    # positional features

    features.update(
        get_positional_features(relative_pos, pop_layout, include_raw=include_raw)
    )

    # content features

    features.update(
        get_content_features(block.boxs, include_raw=include_raw, tag_groups=tag_groups)
    )

    # visual features

    features.update(get_visual_features(block.boxs, include_raw=include_raw))

    # TODO: you can add more features here

    return features
