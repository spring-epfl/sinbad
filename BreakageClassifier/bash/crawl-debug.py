import argparse
import sys
from pathlib import Path
from typing import List
from BreakageClassifier.code.crawl.ablockers import adguard, ublock
from BreakageClassifier.code.crawl.crawl import Crawler, CrawlerConfig
from Saliency.classify import SaliencyClassifierConfig
from Saliency.segment.vips.vips import Vips
from Saliency.utils import saliency_score


def main(program: str, args: List[str]):
    """run the main crawling pipeline

    Args:
        program (str)
        args (List[str])
    """

    parser = argparse.ArgumentParser(
        prog=program, description="Run a crawling experiment"
    )

    parser.add_argument(
        "--issues",
        type=Path,
        help="input issues.",
        default=Path("../forums/ublock/data/ublock-data.csv"),
    )

    parser.add_argument(
        "--filters",
        type=Path,
        help="Filters directory.",
        default=Path("../forums/ublock/data/filterlists/"),
    )

    parser.add_argument(
        "--issue",
        type=str,
        help="issue to debug.",
        default=None,
    )

    parser.add_argument(
        "--sal",
        type=Path,
        help="Saliency Classifier folder path. should contain model.pkl and features.txt",
        default=None,
    )

    parser.add_argument(
        "--out",
        type=Path,
        help="Directory to output the results.",
        default=Path("./datadir"),
    )

    parser.add_argument(
        "--adblocker",
        type=str,
        help="adblocker to use.",
        choices=["ublock", "adguard"],
    )

    parser.add_argument(
        "--headless",
        type=bool,
        help="headless mode.",
        default=False,
    )

    ADBLOCKERS = {
        "ublock": ublock,
        "adguard": adguard,
    }

    ns = parser.parse_args(args)

    # Configuration

    saliency_conf = None

    if ns.sal is not None:
        saliency_conf = SaliencyClassifierConfig(
            fp=ns.sal,
            segment=Vips,
            pre_scoring=saliency_score,
            pre_scoring_threshold=0,
        )

    conf = CrawlerConfig(
        saliency=saliency_conf,
        screenshots=True,
        adblocker=ADBLOCKERS[ns.adblocker],
        dom_dump_timeout=3 * 60,
        filterlist_load_timeout=4 * 60,
        log_debug=True,
        headless=ns.headless,
    )

    crawler = Crawler(
        num_browsers=1,
        data_dir=str(ns.out.absolute()),
        conf=conf,
        forced=True,
    )
    crawler.debug_issue(ns.issues, ns.filters, int(ns.issue))


if __name__ == "__main__":
    main(sys.argv[0], sys.argv[1:])
