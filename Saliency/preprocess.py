from __future__ import annotations
from pathlib import Path
from tabnanny import verbose
from typing import List, Optional, Tuple
from tqdm import tqdm
import json
import pandas as pd
from Saliency.utils import BrowserContext

from Saliency.features import (
    get_block_features,
    update_features_with_population,
)


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



def preprocess_crawl(
    blocks: List[BlockStruct],
    tag_groups={
        "functional": {"a", "img", "button", "iframe", "video", "label"},
        "text": {
            "span",
            "p",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "big",
            "em",
            "font",
            "i",
        },
        "layout": {"table", "tr", "td", "th", "section", "header", "div"},
    },
    browser_context: Optional[BrowserContext] = None,
) -> pd.DataFrame:
    """Preprocesses the blocks into a dataframe of features

    Args:
        blocks (List[BlockStruct]): the list of blocks to preprocess
        tag_groups (dict, optional): the tag groups to use for the features.
    Returns:
        pd.DataFrame: the dataframe of features
    """
    block_features = [
        get_block_features(
            block,
            blocks,
            browser_context,
            tag_groups=tag_groups,
        )
        for block in blocks
    ]

    out_df = pd.DataFrame(block_features)

    out_df = update_features_with_population(out_df, tag_groups)

    return out_df
