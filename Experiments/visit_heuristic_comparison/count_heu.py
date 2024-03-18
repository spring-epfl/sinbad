import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from BreakageClassifier.code.graph.database import Database
from BreakageClassifier.code.run import GraphExtractionError

NODE_TYPES = ["img", "#text", "button"]

# + network requests


def get_diff_resp_counts(responses_in: pd.DataFrame, responses_out: pd.DataFrame):
    urls_in = responses_in.url.unique().tolist()
    urls_out = responses_out.url.unique().tolist()

    only_prev = set(urls_in) - set(urls_out)
    only_new = set(urls_out) - set(urls_in)
    both = set(urls_in) & set(urls_out)

    return {
        "req_total": len(only_prev) + len(only_new) + len(both),
        "req_blocked_new": len(
            only_prev
        ),  # the requests only in the previous version are blocked in the new version
        "req_allowed_new": len(
            only_new
        ),  # the requests only in the new version are allowed in the new version
    }


def get_diff_node_count(
    df_dom_in: pd.DataFrame,
    df_dom_out: pd.DataFrame,
    node_types: list = NODE_TYPES,
):
    counts = {
        "dn_total": len(df_dom_out) - len(df_dom_in),
        "dn_total_imp": 0,
        "dn_total_imp_vis": 0,
    }

    def is_visible(visual_cues):
        try:
            return json.loads(visual_cues)["is_visible"]
        except:
            return False

    for node_type in node_types:
        in_df = df_dom_in[df_dom_in.nodeName == node_type]
        out_df = df_dom_out[df_dom_out.nodeName == node_type]

        n_in = len(in_df)
        n_vis_in = len(in_df[in_df.visual_cues.apply(is_visible)])

        n_out = len(out_df)
        n_vis_out = len(out_df[out_df.visual_cues.apply(is_visible)])

        counts[f"dn_{node_type}"] = n_out - n_in
        counts[f"dn_{node_type}_vis"] = n_vis_out - n_vis_in

        counts[f"r_{node_type}"] = abs(counts[f"dn_{node_type}"]) / n_in if n_in else 0
        counts[f"r_{node_type}_vis"] = (
            abs(counts[f"dn_{node_type}_vis"]) / n_vis_in if n_vis_in else 0
        )

        counts["dn_total_imp"] += counts[f"dn_{node_type}"]
        counts["dn_total_imp_vis"] += counts[f"dn_{node_type}_vis"]

    return counts


def extract_features(
    crawl_dir: Path, output_dir: Path, issues: list = None, label=1, to_csv=True
):
    db_file: Path = crawl_dir / "crawl-data.sqlite"
    exp_file: Path = crawl_dir / "experiments.csv"

    output_dir.mkdir(parents=True, exist_ok=True)

    with Database(db_file, exp_file) as database:
        sites_visits = database.sites_visits()

        if issues is not None:
            sites_visits = sites_visits[
                (sites_visits.issue_id.astype(str).isin(issues))
            ]

        df_dom = database.get_dom()
        df_requests = database.get_requests()

    if len(sites_visits) == 0:
        raise GraphExtractionError

    # after to before -> broken
    # no to after -> not broken
    # before to after -> not broken
    # no to before -> broken

    visit_pairs = [
        ("visit_id_a", "visit_id_b", label),
        # ("visit_id_u", "visit_id_a", -1),
        # ("visit_id_b", "visit_id_a", -1),
        # ("visit_id_u", "visit_id_b", 1),
    ]

    counts = []

    for _, issue in tqdm(sites_visits.iterrows(), total=len(sites_visits)):
        for visit_id_a, visit_id_b, label in visit_pairs:
            _dom_counts = get_diff_node_count(
                df_dom[df_dom.visit_id == issue[visit_id_a]],
                df_dom[df_dom.visit_id == issue[visit_id_b]],
            )
            _req_counts = get_diff_resp_counts(
                df_requests[df_requests.visit_id == issue[visit_id_a]],
                df_requests[df_requests.visit_id == issue[visit_id_b]],
            )
            _dom_counts["visit_id_prev"] = issue[visit_id_a]
            _dom_counts["visit_id_new"] = issue[visit_id_b]
            _dom_counts["issue_id"] = issue.issue_id
            _dom_counts["is_broken"] = label

            counts.append(_dom_counts | _req_counts)

    counts = pd.DataFrame(counts)

    if to_csv:
        counts.to_csv(output_dir / f"count_heu_features_{label}.csv", index=False)

    return counts


def extract_features_mult(crawl_dirs: list, output_dir: Path):
    features = []

    for crawl_dir, label in crawl_dirs:
        features.append(extract_features(crawl_dir, output_dir, label=label))

    # align columns
    for i in range(1, len(features)):
        features[i] = features[i][features[0].columns]

    features = pd.concat(features)

    features.to_csv(output_dir / "count_heu_features.csv", index=False)

    return features


class CountsThresholdHeuristic:
    def __init__(self, features, threshold, data):
        self.features = features
        self.threshold = threshold
        self.data = data

    def predict(self, issue):
        x = self.data[
            (self.data.issue_id == issue)
        ]

        x = x.iloc[0]

        if any(abs(x[f]) >= self.threshold for f in self.features):
            return 1

        return -1


class RatioThresholdHeuristic:
    def __init__(self, features, data, threshold=0.5):
        self.features = features
        self.data = data
        self.threshold = threshold

    def predict_proba(self, issue):
        x = self.data[
            (self.data.issue_id == issue)
        ].iloc[0]

        ratio = sum(abs(x[f]) for f in self.features) / len(self.features)

        return ratio

    def predict(self, issue):
        return 1 if self.predict_proba(issue) >= self.threshold else -1
