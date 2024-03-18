import logging
import pickle
from re import S
import time
from typing import Callable, List, Optional
import numpy as np
import pandas as pd
from Saliency.segment.vips.block_vo import BlockVo
from Saliency.segment.vips.vips import Vips
from zipp import Path
from Saliency.preprocess import preprocess_crawl
from Saliency.utils import (
    BrowserContext,
    DomEncoder,
    SegmentationGenerator,
    pre_score_blocks,
)


class SaliencyClassifierConfig:
    def __init__(
        self,
        fp: Path,
        segment: SegmentationGenerator = Vips,
        pre_scoring: Optional[
            Callable[[List[BlockVo], BrowserContext], List[BlockVo]]
        ] = None,
        pre_scoring_threshold: float = 0,
    ):
        self.fp = fp
        self.segmentor = segment
        self.pre_scoring_method = pre_scoring
        self.pre_scoring_threshold = pre_scoring_threshold


class SaliencyClassifier:
    class NoBlocksFound(Exception):
        pass

    def __init__(
        self,
        conf: SaliencyClassifierConfig,
        logger: logging.Logger,
    ):
        self.fp = conf.fp
        self.logger = logger
        self.segmentor = conf.segmentor
        self.pre_scoring_method = conf.pre_scoring_method
        self.pre_scoring_threshold = conf.pre_scoring_threshold

        with open(conf.fp / "model.pkl", "rb") as f:
            self.model = pickle.load(f)

        with open(conf.fp / "features.txt", "r") as f:
            self.features = f.read().splitlines()

    def set_logger(self, logger: logging.Logger):
        self.logger = logger

    def predict_features(self, input: pd.DataFrame) -> np.ndarray:
        """Predicts the saliency of the input features

        Args:
            input (pd.DataFrame): the input features dataframe. saliencyClassifier.features should be a subset of the columns

        Returns:
            np.ndarray: the predicted saliency (n_samples, 1)
        """

        test_mani = input[self.features].copy()
        X = test_mani.to_numpy()

        return self.model.predict(X)

    def predict_blocks(self, blocks: List[BlockVo], browser_context: BrowserContext):
        """Predicts the saliency of the blocks in the list

        Args:
            blocks (List[BlockVo]): the list of blocks to predict from the segmentation algorithm
            browser_context (BrowserContext): the browser invariants, from a selenium.webdriver

        Raises:
            SaliencyClassifer.NoBlocksFound: if no blocks are found

        Returns:
            dict[str, int]: the predicted saliency for each block in the list
                example:
                ```
                    {
                        89: 1,
                        28: 0,
                        ...
                    }
                ```
        """

        if len(blocks) == 0:
            raise SaliencyClassifier.NoBlocksFound("No salient elements found")

        if self.pre_scoring_method is not None:
            t_start = time.time()
            blocks, _ = pre_score_blocks(
                blocks,
                browser_context,
                self.pre_scoring_method,
                self.pre_scoring_threshold,
            )
            self.logger.debug(
                "Pre-scoring finished in %f seconds", time.time() - t_start
            )

        t_start = time.time()
        bf = preprocess_crawl(
            blocks,
            browser_context=browser_context,
        )
        self.logger.debug("Preprocessing finished in %f seconds", time.time() - t_start)

        t_start = time.time()

        labels = self.predict_features(bf)

        self.logger.debug("Prediction finished in %f seconds", time.time() - t_start)

        return {b.id: int(labels[i]) for i, b in enumerate(blocks)}

    def predict_json(self, nodes_json: dict, browser_context: BrowserContext):
        """Predicts the saliency of the blocks in the json. The json is generated from the webpage after running `dom.js`

        Args:
            nodes_json (dict): the json representation of the webpage
                example:
                ```
                    {
                        nodetype: ...,
                        ...,
                        childNodes: [...]
                    }
                ```
            browser_context(BrowserContext): the browser invariants, from a selenium.webdriver

        Raises:
            SaliencyClassifer.NoBlocksFound: if no blocks are found

        Returns:
            List[BlockVO], dict[str, int]: the predicted saliency for each block in the list
                example:
                ```
                [...],
                {
                    89: 1,
                    28: 0,
                    ...
                }
                ```
        """
        # encode into List of DomNode(s)
        encoder = DomEncoder(self.logger)

        t_start = time.time()

        self.logger.debug("Encoding DOM")

        parent_node = encoder.to_dom(nodes_json)

        self.logger.debug("DOM encoded in %f seconds", time.time() - t_start)

        t_start = time.time()
        # segment
        block_list = self.segmentor(encoder.nodeList, browser_context).service()

        self.logger.debug("DOM segmented in %f seconds", time.time() - t_start)

        # check list
        if (
            block_list is None
            or not isinstance(block_list, list)
            or len(block_list) == 0
        ):
            self.logger.info("Found NO salient elements")
            raise SaliencyClassifier.NoBlocksFound("No salient elements found")

        return self.predict_blocks(block_list, browser_context), block_list, parent_node
