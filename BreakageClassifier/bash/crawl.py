import argparse
import sys
from pathlib import Path
from typing import List
from BreakageClassifier.code.crawl.crawl import Crawler, CrawlerConfig
from Saliency.classify import SaliencyClassifierConfig
from Saliency.segment.vips.vips import Vips
from Saliency.utils import saliency_score
from BreakageClassifier.code.crawl.ablockers import ublock, adguard
from openwpm.errors import CommandExecutionError, BrowserCrashError


def main(program: str, args: List[str]):
    """run the main crawling pipeline

    Args:
        program (str)
        args (List[str])
    """

    parser = argparse.ArgumentParser(
        prog=program, description="Run a crawling experiment"
    )

    parser.add_argument("--n", type=int, help="sample number", default=None)
    parser.add_argument(
        "--issues",
        type=Path,
        help="input issues.",
        default=Path("../forums/ublock/data/ublock-data.csv"),
    )
    parser.add_argument(
        "--ignore",
        type=Path,
        help="ignore issues.",
        default=None,
    )
    parser.add_argument(
        "--filters",
        type=Path,
        help="Filters directory.",
        default=Path("../forums/ublock/data/filterlists/"),
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
        "--start",
        type=int,
        help="the index of the sample to start from",
        default=0,
    )

    parser.add_argument(
        "--f",
        type=bool,
        help="Force reset the crawl in this folder",
        default=0,
    )

    parser.add_argument(
        "--adblocker",
        type=str,
        help="adblocker to use.",
        choices=["ublock", "adguard"],
    )

    parser.add_argument(
        "--tryforever",
        type=bool,
        help="try forever to crawl sites",
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
        adblocker=ADBLOCKERS[ns.adblocker],
        filterlist_load_timeout=4 * 120,
    )

    if not ns.tryforever:
        crawler = Crawler(
            num_browsers=1,
            data_dir=str(ns.out.absolute()),
            conf=conf,
            forced=ns.f,
        )
        crawler.crawl_from_dataset(
            ns.issues,
            ns.filters,
            num=ns.n,
            start=ns.start,
            ignore=ns.ignore,
        )

    else:
        failed_issues = []
        session_num = 0
        last_failed_issue = None

        while True:
            try:
                crawler = Crawler(
                    num_browsers=1,
                    data_dir=str(ns.out.absolute() / f"session_{session_num}"),
                    conf=conf,
                    forced=ns.f,
                )
                crawler.crawl_from_dataset(
                    ns.issues,
                    ns.filters,
                    num=ns.n,
                    start=ns.start,
                    ignore=ns.ignore,
                    start_after_issue=last_failed_issue,
                )
            except (CommandExecutionError, BrowserCrashError) as e:
                print("Browser crashed retrying...")
                session_num += 1
                failed_issues.append(crawler._curr_issue)
                last_failed_issue = crawler._curr_issue
                continue
            break

        with open(ns.out.absolute() / f"crash_issues.txt", "w") as f:
            f.write("\n".join(failed_issues))


if __name__ == "__main__":
    main(sys.argv[0], sys.argv[1:])
