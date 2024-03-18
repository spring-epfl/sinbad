import argparse
import sys
import traceback
from pathlib import Path
from typing import Iterable, List, Optional, Union
import time

import pandas as pd
from scipy import stats
from tqdm import tqdm
from yaml import full_load
from BreakageClassifier.code.features.constants import (
    EDGES_DF_COLUMNS_LABELED,
    EDITS_DF_COLUMNS,
    EDITS_DF_COLUMNS_LABELED,
    NODES_DF_COLUMNS_LABELED,
)
from BreakageClassifier.code.features.dataflow import (
    get_dom_interactive_features,
    get_global_features,
)
from BreakageClassifier.code.features.structure import get_structure_features
from BreakageClassifier.code.features.utils import (
    add_missing_cols,
    get_subtree,
    index_subtrees,
    mix_features,
    standarize_id,
    time_it,
    visualize_edit_tree,
    visualize_tree,
)

from BreakageClassifier.code.graph.html_edges import extract_js_edges
from .graph.requests import augment_dom_with_requests
from .graph.error import augment_dom_with_interactions
from .graph import constants as NODE_TYPES
import leveldb

from storage_dump.storage import DataframeCSVStorageController

from . import graph as gs
from .graph.database import Database
from . import labelling as ls
from .labelling import filterlists as fs
from .labelling import dom as dl

from .features.feature_extraction import extract_graph_features
from .features.dom import (
    add_saliencies_to_tree,
    clean_tree,
    compare_graphs,
    compare_trees_parallel,
    parse_attributes,
)
from .features.content import get_dom_edits_content_features

from .utils import return_none_if_fail
from .logger import LOGGER

pd.set_option("display.max_rows", None, "display.max_columns", None)

__dir__ = Path(__file__).parent


def load_config_info(filename: str) -> dict:
    """Load features from features.yaml file
    :param filename: yaml file name containing feature names
    :return: dict of features to use.
    """
    with open(filename) as file:
        return full_load(file)


def extract_features(
    pdf: pd.DataFrame, networkx_graph, visit_id: str, config_info: dict, ldb_file: str
) -> pd.DataFrame:
    """Getter to generate the features of each node in a graph.
    :param pdf: pandas df of nodes and edges in a graph.
    :param G: Graph object representation of the pdf.
    :return: dataframe of features per node in the graph
    """
    # Generate features for each node in our graph
    ldb = leveldb.LevelDB(ldb_file)
    df_features = extract_graph_features(
        pdf, networkx_graph, visit_id, ldb, config_info
    )
    return df_features


@return_none_if_fail()
def find_setter_domain(setter: str) -> str:
    """Finds the domain from a setter
    :param setter: string setter value
    :return: string domain value
    """
    domain = gs.get_domain(setter)
    return domain


@return_none_if_fail(is_debug=True)
def find_domain(row: pd.Series) -> Union[str, None]:
    """Finds the domain of a node
    :param row: a row from the graph df representing a node.
    :return: string domain value or none if N/A
    """
    domain = None
    node_type = row["type"]
    if node_type == NODE_TYPES.REQUEST_TYPE:
        domain = gs.get_domain(gs.get_val_from_attributes(row.attributes, "url"))
    else:
        return domain
    return domain


@return_none_if_fail()
def find_tld(top_level_url: str) -> Union[str, None]:
    """Finds the top level domain from a top level url
    :param top_level_url: string of the url
    :return: string domain value or none if N/A
    """
    if top_level_url:
        tld = gs.get_domain(top_level_url)
        return tld
    else:
        return None


def get_party(row: pd.Series) -> str:
    """Finds whether a storage node is first party or third party
    :param row: a row from the graph df representing a node.
    :return: string party (first | third | N/A)
    """
    if row["domain"] and row["top_level_domain"]:
        if row["domain"] == row["top_level_domain"]:
            return "first"
        else:
            return "third"
    return "N/A"


def find_setters(
    df_all_storage_nodes: pd.DataFrame,
    df_http_cookie_nodes: pd.DataFrame,
    df_all_storage_edges: pd.DataFrame,
    df_http_cookie_edges: pd.DataFrame,
) -> pd.DataFrame:
    """Finds the nodes that first set each of the present cookies.
    :param row: a row from the graph df representing a node.
    :return: string domain value or none if N/A
    """

    df_setter_nodes = pd.DataFrame(
        columns=[
            "visit_id",
            "name",
            "type",
            "attr",
            "top_level_url",
            "domain",
            "setter",
            "setting_time_stamp",
        ]
    )

    try:
        df_storage_edges = pd.concat([df_all_storage_edges, df_http_cookie_edges])
        if len(df_storage_edges) > 0:
            # get all set events for http and js cookies
            df_storage_sets = df_storage_edges[
                (df_storage_edges["action"] == "set")
                | (df_storage_edges["action"] == "set_js")
            ]

            # find the initial setter nodes for each cookie
            df_setters = gs.get_original_cookie_setters(df_storage_sets)

            df_storage_nodes = pd.concat([df_all_storage_nodes, df_http_cookie_nodes])
            df_setter_nodes = df_storage_nodes.merge(
                df_setters, on=["visit_id", "name"], how="outer"
            )

    except Exception as e:
        LOGGER.warning("Error getting setter", exc_info=True)

    return df_setter_nodes


@time_it
def build_graph(
    database: Database,
    visit_id: str,
    site_url: str,
    salient_nodes: Optional[pd.DataFrame] = None,
):
    """Read SQL data from crawler for a given visit_ID.
    :param visit_id: visit ID of a crawl URL.
    :return: Parsed information (nodes and edges) in pandas df.
    """

    """
    # Read tables from DB and store as DataFrames
    df_requests, df_responses, df_redirects, call_stacks, javascript = database.website_from_visit_id(visit_id)
    
    # extract nodes and edges from all categories of interest as described in the paper
    df_js_nodes, df_js_edges = gs.build_html_components(javascript)
    df_request_nodes, df_request_edges = gs.build_request_components(df_requests, df_responses, df_redirects, call_stacks)
    df_all_storage_nodes, df_all_storage_edges = gs.build_storage_components(javascript)
    df_http_cookie_nodes, df_http_cookie_edges = gs.build_http_cookie_components(df_request_edges, df_request_nodes)
    df_storage_node_setter = find_setters(df_all_storage_nodes, df_http_cookie_nodes, df_all_storage_edges, df_http_cookie_edges)

    # Concatenate to get all nodes and edges
    df_request_nodes['domain'] = None
    df_all_nodes = pd.concat([df_js_nodes, df_request_nodes, df_storage_node_setter])
    df_all_nodes['domain'] = df_all_nodes.apply(find_domain, axis=1)
    df_all_nodes['top_level_domain'] = df_all_nodes['top_level_url'].apply(find_tld)
    df_all_nodes['setter_domain'] = df_all_nodes['setter'].apply(find_setter_domain)
    df_all_nodes = df_all_nodes.drop_duplicates()
    df_all_nodes['graph_attr'] = "Node"

    df_all_edges = pd.concat([df_js_edges, df_request_edges, df_all_storage_edges, df_http_cookie_edges])
    df_all_edges = df_all_edges.drop_duplicates()
    df_all_edges['top_level_domain'] = df_all_edges['top_level_url'].apply(find_tld)
    df_all_edges['graph_attr'] = "Edge"

    #Remove all non-FP cookies, comment for unblocked
    df_all_nodes['party'] = df_all_nodes.apply(get_party, axis=1)
    third_parties = df_all_nodes[df_all_nodes['party'] == 'third']['name'].unique()
    df_all_nodes = df_all_nodes[~df_all_nodes['name'].isin(third_parties)]
    df_all_edges = df_all_edges[~df_all_edges['dst'].isin(third_parties)]
    df_all_edges = df_all_edges[~df_all_edges['src'].isin(third_parties)]

    df_all_graph = pd.concat([df_all_nodes, df_all_edges])
    df_all_graph = df_all_graph.astype(
        {
            'type' : 'category',
            'response_status' : 'category'
        }
    )

    return df_all_graph

    """

    # print("!starting build graph")

    df_dom = database.get_dom_from_visit_id(visit_id)

    # remove unnecessary nodes
    df_dom = clean_tree(df_dom)

    # print("!got dom")

    # if we have provided saliencies we need to encode those
    if salient_nodes is not None:
        df_dom = add_saliencies_to_tree(df_dom, salient_nodes)

    # print("!got saliency. getting js and responses")

    df_responses = database.get_http_responses(visit_id)
    df_javascript = database.get_javascript_events(visit_id)
    df_callstacks = database.get_callstacks(visit_id)

    # print("!got js and responses. getting interaction logs")

    # interactions, interaction_errors = database.get_interaction_logs(visit_id)
    interactions, df_javascript, df_responses = database.get_interaction_logs_all(
        visit_id, df_javascript, df_responses
    )

    # print("interactions=", len(interactions))
    # print("df_javascript=", len(df_javascript))
    # print("df_responses=", len(df_responses))

    # print("!got interaction logs. augmenting with interactions")

    if not interactions.empty:
        df_dom = augment_dom_with_interactions(df_dom, interactions)
    # else:
    #     print("!no interactions")

    # print("!augmented with interactions. augmenting with responses")

    df_dom = augment_dom_with_requests(df_dom, df_responses)

    # print("!augmented with responses. extracting js edges")

    df_nodes, df_edges = extract_js_edges(
        df_dom, df_javascript, interactions, df_callstacks, df_responses
    )

    # print("!extracted js edges")

    df_dom["domain"] = df_dom.apply(find_domain, axis=1)
    df_dom["top_level_domain"] = find_tld(site_url)
    df_dom["party"] = df_dom.apply(get_party, axis=1)

    df_dom["id"] = df_dom["id"].apply(standarize_id)
    df_dom["parent_id"] = df_dom["parent_id"].apply(standarize_id)

    # replace nan with None
    df_dom = df_dom.where(pd.notnull(df_dom), None)

    df_dom[["attr_id", "attr_class", "attr_src"]] = df_dom.apply(
        parse_attributes, axis=1, result_type="expand"
    )

    return df_dom, df_nodes, df_edges


def apply_tasks(
    df: pd.DataFrame,
    visit_id: int,
    config_info: dict,
    ldb_file: Path,
    output_dir: Path,
    overwrite: bool,
    filterlists: list[str],
    filterlist_rules: dict,
):
    """Sequence of tasks to apply on each website crawled.
    :param df: the graph data (nodes and edges) in pandas df.
    :param visit_id: visit ID of a crawl URL.
    :param config_info: dictionary containing features to use.
    :param ldb_file: path to ldb file.
    :param output_dir: path to the output directory.
    :param overwrite: set True to overwrite the content of the output directory.
    """

    # Build the graph
    LOGGER.info("%s %d %d", df.iloc[0]["top_level_url"], visit_id, len(df))
    graph_columns = config_info["graph_columns"]
    feature_columns = config_info["feature_columns"]
    label_columns = config_info["label_columns"]

    try:
        start = time.time()

        # export the graph dataframe to csv file
        graph_path = output_dir / "graph.csv"
        if overwrite or not graph_path.is_file():
            df.reindex(columns=graph_columns).to_csv(str(graph_path))
        else:
            df.reindex(columns=graph_columns).to_csv(
                str(graph_path), mode="a", header=False
            )

        # building networkx_graph
        networkx_graph = gs.build_networkx_graph(df)

        # extracting features from graph data and graph structure
        df_features = extract_features(
            df, networkx_graph, visit_id, config_info, ldb_file
        )
        features_path = output_dir / "features.csv"

        # export the features to csv file
        if overwrite or not features_path.is_file():
            df_features.reindex(columns=feature_columns).to_csv(str(features_path))
        else:
            df_features.reindex(columns=feature_columns).to_csv(
                str(features_path), mode="a", header=False
            )
        end = time.time()
        LOGGER.info("Extracted features: %d", end - start)

        # Label data
        df_labelled = ls.label_data(df, filterlists, filterlist_rules)
        if len(df_labelled) > 0:
            labels_path = output_dir / "labelled.csv"
            if overwrite or not labels_path.is_file():
                df_labelled.reindex(columns=label_columns).to_csv(str(labels_path))
            else:
                df_labelled.reindex(columns=label_columns).to_csv(
                    str(labels_path), mode="a", header=False
                )

    except Exception as e:
        LOGGER.warning("Errored in pipeline", exc_info=True)


@time_it
def process_differential_experiment(
    visit_data_in,
    visit_data_out,
    browser_id: str,
):
    # For each visit, grab the visit_id

    visit_in, (dom_in, nodes_in, edges_in) = visit_data_in
    visit_out, (dom_out, nodes_out, edges_out) = visit_data_out

    # TODO: add the differences in nodes and edges
    # nodes added, removed, edited
    # edges added, removed, edited

    common_subtree, altered_subtrees = compare_trees_parallel(dom_in, dom_out)

    altered_subtrees["other_id"] = altered_subtrees.other_id.astype("object").apply(
        standarize_id
    )
    altered_subtrees["other_parent_id"] = altered_subtrees.other_parent_id.astype(
        "object"
    ).apply(standarize_id)

    # print("!compared trees. comparing graphs")

    add_missing_cols(altered_subtrees, EDITS_DF_COLUMNS)

    nodes, edges = compare_graphs(
        common_subtree,
        altered_subtrees,
        nodes_in,
        edges_in,
        nodes_out,
        edges_out,
        browser_id,
        visit_in,
        browser_id,
        visit_out,
    )

    return (
        dom_in[dom_in.saliency == 1],
        dom_out[dom_out.saliency == 1],
        common_subtree,
        altered_subtrees.astype(
            {
                "visit_id": "object",
                "id": "object",
                "other_visit_id": "object",
                "other_id": "object",
            }
        ),
        nodes,
        edges,
    )


class GraphExtractionError(Exception):
    pass


@time_it
def process_issue(
    issue: pd.Series, database: Database, storage: DataframeCSVStorageController
):
    issue_log = {
        "issue_id": issue.issue_id,
        "no_to_fixed_time": 0,
        "fixed_to_broken_time": 0,
        "no_to_broken_time": 0,
        "subtree_count": 0,
        "node_count": 0,
        "graph_node_count": 0,
        "graph_edge_count": 0,
        "neutral_subtree_count": 0,
        "broken_subtree_count": 0,
        "fixed_subtree_count": 0,
        "after_graph_time": 0,
        "before_graph_time": 0,
        "nofilter_graph_time": 0,
        "label_subtrees_time": 0,
        "success": True,
        "error": None,
    }

    _edits = None
    df_nodes = None
    df_edges = None
    salient_nodes = None

    try:
        LOGGER.info("Building graphs for issue %s", issue.issue_id)
        # build graphs
        LOGGER.debug("Building after graph")
        t, graphs_after = build_graph(database, issue.visit_id_a, issue.site_url)
        storage.save(
            "stats-extract.csv",
            {
                "issue": issue.issue_id,
                "function": "build_graph:after",
                "time": t,
            },
        )
        issue_log["after_graph_time"] = t

        dom_after = graphs_after[0]
        salient_nodes = dom_after[dom_after.saliency == 1.0].copy()

        t, graphs_before = build_graph(
            database, issue.visit_id_b, issue.site_url, salient_nodes
        )
        storage.save(
            "stats-extract.csv",
            {
                "issue": issue.issue_id,
                "function": "build_graph:before",
                "time": t,
            },
        )
        issue_log["before_graph_time"] = t

        t, graphs_nofilter = build_graph(
            database, issue.visit_id_u, issue.site_url, salient_nodes
        )
        storage.save(
            "stats-extract.csv",
            {
                "issue": issue.issue_id,
                "function": "build_graph:nofilter",
                "time": t,
            },
        )
        issue_log["nofilter_graph_time"] = t

        # build differential trees
        LOGGER.info("Building diff-trees")
        LOGGER.debug("Building no-to-fixed diff-tree")

        t, (
            *_,
            salient_nodes,
            no_to_fixed_common,
            no_to_fixed_diff,
            no_to_fixed_nodes,
            no_to_fixed_edges,
        ) = process_differential_experiment(
            (issue.visit_id_u, graphs_nofilter),
            (issue.visit_id_a, graphs_after),
            issue.browser_id_u,
        )

        salient_nodes["issue"] = issue.issue_id

        storage.save(
            "stats-extract.csv",
            {
                "issue": issue.issue_id,
                "function": "process_differential_experiment:no_to_fixed",
                "time": t,
            },
        )
        issue_log["no_to_fixed_time"] = t

        LOGGER.debug("Building fixed-to-broken diff-tree")

        t, (
            *_,
            fixed_to_broken_diff,
            fixed_to_broken_nodes,
            fixed_to_broken_edges,
        ) = process_differential_experiment(
            (issue.visit_id_a, graphs_after),
            (issue.visit_id_b, graphs_before),
            issue.browser_id_u,
        )

        storage.save(
            "stats-extract.csv",
            {
                "issue": issue.issue_id,
                "function": "process_differential_experiment:fixed_to_broken",
                "time": t,
            },
        )
        issue_log["fixed_to_broken_time"] = t

        LOGGER.debug("Building no-to-broken diff-tree")
        t, (
            *_,
            no_to_broken_diff,
            no_to_broken_nodes,
            no_to_broken_edges,
        ) = process_differential_experiment(
            (issue.visit_id_u, graphs_nofilter),
            (issue.visit_id_b, graphs_before),
            issue.browser_id_u,
        )

        storage.save(
            "stats-extract.csv",
            {
                "issue": issue.issue_id,
                "function": "process_differential_experiment:no_to_broken",
                "time": t,
            },
        )
        issue_log["no_to_broken_time"] = t

        # get the differential trees with labels

        LOGGER.info("Labeling subtrees")
        t, _edits = dl.label_alterations(
            issue,
            no_to_fixed_common,
            no_to_fixed_diff,
            fixed_to_broken_diff,
            no_to_broken_diff,
        )

        storage.save(
            "stats-extract.csv",
            {
                "issue": issue.issue_id,
                "function": "label_alterations",
                "time": t,
            },
        )
        issue_log["label_subtrees_time"] = t

        # generate the contexts

        no_to_fixed_nodes["origin"] = "no_fixed"
        no_to_fixed_nodes["flipped"] = False
        no_to_fixed_nodes.rename(
            columns={"rem_in_l": "added", "rem_in_r": "removed"}, inplace=True
        )

        no_to_fixed_edges["origin"] = "no_fixed"
        no_to_fixed_edges["flipped"] = False
        no_to_fixed_edges.rename(
            columns={"rem_in_l": "added", "rem_in_r": "removed"}, inplace=True
        )

        fixed_to_broken_nodes["origin"] = "fixed_broken"
        fixed_to_broken_nodes["flipped"] = False
        fixed_to_broken_nodes.rename(
            columns={"rem_in_l": "added", "rem_in_r": "removed"}, inplace=True
        )

        fixed_to_broken_edges["origin"] = "fixed_broken"
        fixed_to_broken_edges["flipped"] = False
        fixed_to_broken_edges.rename(
            columns={"rem_in_l": "added", "rem_in_r": "removed"}, inplace=True
        )

        no_to_broken_nodes["origin"] = "no_broken"
        no_to_broken_nodes["flipped"] = False
        no_to_broken_nodes.rename(
            columns={"rem_in_l": "added", "rem_in_r": "removed"}, inplace=True
        )

        no_to_broken_edges["origin"] = "no_broken"
        no_to_broken_edges["flipped"] = False
        no_to_broken_edges.rename(
            columns={"rem_in_l": "added", "rem_in_r": "removed"}, inplace=True
        )

        # only fixed_broken has fipping
        fixed_to_broken_nodes_flipped = fixed_to_broken_nodes.copy()

        fixed_to_broken_nodes_flipped["flipped"] = True
        fixed_to_broken_nodes.rename(
            columns={"removed": "added", "added": "removed"}, inplace=True
        )

        fixed_to_broken_edges_flipped = fixed_to_broken_edges.copy()

        fixed_to_broken_edges_flipped["flipped"] = True
        fixed_to_broken_edges_flipped.rename(
            columns={"removed": "added", "added": "removed"}, inplace=True
        )

        df_nodes = pd.concat(
            [
                no_to_fixed_nodes,
                fixed_to_broken_nodes,
                no_to_broken_nodes,
                fixed_to_broken_nodes_flipped,
            ]
        )

        df_nodes["issue"] = issue.issue_id

        df_edges = pd.concat(
            [
                no_to_fixed_edges,
                fixed_to_broken_edges,
                no_to_broken_edges,
                fixed_to_broken_edges_flipped,
            ]
        )

        df_edges["issue"] = issue.issue_id

        if not _edits.empty:
            issue_log["subtree_count"] = len(_edits[_edits["is_root"] == True])
            issue_log["node_count"] = len(_edits)

            edit_stat = (
                _edits[_edits["is_root"]][["is_breaking"]].value_counts().to_dict()
            )

            issue_log["neutral_subtree_count"] = edit_stat.get((0,), 0)
            issue_log["broken_subtree_count"] = edit_stat.get((1,), 0)
            issue_log["fixed_subtree_count"] = edit_stat.get((-1,), 0)

            LOGGER.debug(
                "Generated edits: %i subtrees, %i subtree nodes",
                issue_log["subtree_count"],
                issue_log["node_count"],
            )

            LOGGER.debug(
                "Subtrees distribution: fixed (-1) - %i, neutral (0) - %i, broken (1) - %i",
                issue_log["fixed_subtree_count"],
                issue_log["neutral_subtree_count"],
                issue_log["broken_subtree_count"],
            )

            issue_log["graph_node_count"] = len(df_nodes)
            issue_log["graph_edge_count"] = len(df_edges)

            LOGGER.debug(
                "Generated a global graph of %i nodes and %i edges",
                issue_log["graph_node_count"],
                issue_log["graph_edge_count"],
            )

        else:
            LOGGER.warning("No edits were generated!!")

    except Exception as e:
        issue_log["success"] = False
        issue_log["error"] = str(e)
        # number_failures += 1
        # tqdm.write(f"Fail: {number_failures}")
        # tqdm.write(f"Error: {e}")
        traceback.print_exc()

    return (
        _edits,
        issue_log,
        df_nodes,
        df_edges,
        salient_nodes,
    )


def extract_graphs(
    crawl_dir: Path,
    output_dir: Path,
    issues: Optional[Iterable[int]] = None,
    n_useful_threshold: int = None,
):
    db_file: Path = crawl_dir / "crawl-data.sqlite"
    exp_file: Path = crawl_dir / "experiments.csv"
    number_failures = 0
    edits = []
    env_nodes = []
    env_edges = []

    # setup
    # config_info = load_config_info(features_file)

    output_dir.mkdir(parents=True, exist_ok=True)

    with Database(db_file, exp_file) as database, DataframeCSVStorageController(
        output_dir,
        [
            "edits.csv",
            "log.csv",
            "nodes.csv",
            "edges.csv",
            "stats-extract.csv",
            "salient.csv",
        ],
        columns={
            "edits.csv": EDITS_DF_COLUMNS_LABELED,
            "nodes.csv": NODES_DF_COLUMNS_LABELED,
            "edges.csv": EDGES_DF_COLUMNS_LABELED,
            "salient.csv": [
                "browser_id",
                "visit_id",
                "id",
                "nodeName",
                "type",
                "attributes",
                "visual_cues",
                "parent_id",
                "t_enter",
                "t_leave",
                "saliency",
                "block",
                "issue",
            ],
            "stats-extract.csv": ["issue", "function", "time"],
        },
    ) as out_storage:
        # read site visits
        sites_visits = database.sites_visits()

        LOGGER.info("Found %i sites successfully crawled", len(sites_visits))

        if issues is not None:
            sites_visits = sites_visits[
                (sites_visits.issue_id.astype(str).isin(issues))
            ]

            if len(sites_visits) == 0:
                raise GraphExtractionError(
                    "No sites to process, make sure you have the correct issue IDs"
                )

        n_useful = 0

        salient_nodes = []

        for _, issue in tqdm(
            sites_visits.iterrows(),
            total=len(sites_visits),
            position=0,
            leave=True,
            ascii=True,
        ):
            LOGGER.debug("PROCESSING ISSUE %s", issue.issue_id)
            t, (
                _edits,
                issue_log,
                _env_nodes,
                _env_edges,
                _salient_nodes,
            ) = process_issue(issue, database, out_storage)

            out_storage.save(
                "stats-extract.csv",
                {
                    "issue": issue.issue_id,
                    "function": "process_issue",
                    "time": t,
                },
            )

            if issue_log["success"]:
                if not _edits.empty:
                    edits.append(_edits)
                    out_storage.save("edits.csv", _edits)

                if not _env_nodes.empty:
                    env_nodes.append(_env_nodes)
                    out_storage.save("nodes.csv", _env_nodes)

                if not _env_edges.empty:
                    env_edges.append(_env_edges)
                    out_storage.save("edges.csv", _env_edges)

                if not _salient_nodes.empty:
                    salient_nodes.append(_salient_nodes)
                    out_storage.save("salient.csv", _salient_nodes)

            else:
                number_failures += 1
                tqdm.write(f"Fail: {number_failures} total failed extractions ")

            if issue_log["broken_subtree_count"] > 0:
                n_useful += 1

            out_storage.save("log.csv", pd.DataFrame([issue_log]))

            if n_useful_threshold is not None and n_useful >= n_useful_threshold:
                break

    percent = (number_failures / len(sites_visits)) * 100
    LOGGER.info(
        f"Fail: {number_failures}, Total: {len(sites_visits)}, Percentage:{percent} %s",
        str(db_file),
    )

    return (
        pd.concat(edits).drop_duplicates(),
        pd.concat(env_nodes),
        pd.concat(env_edges),
        pd.concat(salient_nodes),
    )


def load_graphs(
    edits_dir: Path,
):
    return (
        pd.read_csv(edits_dir / "edits.csv"),
        pd.read_csv(edits_dir / "nodes.csv"),
        pd.read_csv(edits_dir / "edges.csv"),
        pd.read_csv(edits_dir / "salient.csv"),
    )


def pipeline(
    crawl_dir: Path,
    # ldb_file: Path,
    # filterlist_dir: Path,
    output_dir: Path,
    features_file: Path = __dir__.joinpath("features.yaml"),
    # overwrite=True,
    edits_dir: Path = None,
    issues: Optional[Iterable[int]] = None,
    n_useful_threshold: int = None,
):
    """Graph processing and labeling pipeline
    :param db_file: the graph data (nodes and edges) in pandas df.
    :param visit_id: visit ID of a crawl URL.
    :param config_info: dictionary containing features to use.
    :param ldb_file: path to ldb file.
    :param output_dir: path to the output directory.
    :param overwrite: set True to overwrite the content of the output directory.
    """

    # if we did not provide an edits directory, we need to extract the graphs
    if edits_dir is None:
        edits, env_nodes, env_edges, salient_nodes = extract_graphs(
            crawl_dir, output_dir, issues, n_useful_threshold=n_useful_threshold
        )

    else:
        edits, env_nodes, env_edges, salient_nodes = load_graphs(edits_dir)

    if edits.empty:
        LOGGER.warning("No edits were generated!!")
        return

    LOGGER.info("Extracted %i edit nodes", len(edits))

    with DataframeCSVStorageController(
        output_dir,
        ["stats-features.csv"],
        columns={
            "stats-features.csv": ["function", "time"],
        },
    ) as out_storage:
        LOGGER.debug("Indexing subtrees")
        t, edits = index_subtrees(edits)
        out_storage.save(
            "stats-features.csv",
            {
                "function": "index_subtrees",
                "time": t,
            },
        )

        # feature extraction from edits
        LOGGER.debug("Extracting content features")
        t, content_features = get_dom_edits_content_features(edits, salient_nodes)
        out_storage.save(
            "stats-features.csv",
            {
                "function": "get_dom_edits_content_features",
                "time": t,
            },
        )

        # feature extraction from structure
        LOGGER.debug("Extracting structural features")
        t, structural_features = get_structure_features(edits)
        out_storage.save(
            "stats-features.csv",
            {
                "function": "get_structure_features",
                "time": t,
            },
        )

        # feature extraction from function
        LOGGER.debug("Extracting functional features")
        t, functional_features = get_dom_interactive_features(
            edits, env_nodes, env_edges
        )
        out_storage.save(
            "stats-features.csv",
            {
                "function": "get_dom_interactive_features",
                "time": t,
            },
        )

        # mix features
        LOGGER.debug("Mixing features")
        t, features = mix_features(
            content_features, structural_features, functional_features
        )
        out_storage.save(
            "stats-features.csv",
            {
                "function": "mix_features",
                "time": t,
            },
        )

        # feature extraction from global
        LOGGER.debug("Extracting global features")
        t, features = get_global_features(features, env_nodes, env_edges)
        out_storage.save(
            "stats-features.csv",
            {
                "function": "get_global_features",
                "time": t,
            },
        )

    features.to_csv(output_dir / "features.csv", index=False)


def _index_subtrees_debug(edits_df):
    LOGGER.debug("Indexing subtrees")

    edits_df["subtree_index"] = -1
    roots = edits_df[edits_df.is_root]

    i = 0

    for _, root in tqdm(roots.iterrows(), total=len(roots)):
        edits_df.loc[
            (
                bool(root.visit_id)
                & ~edits_df["t_enter"].isna()
                & (edits_df["visit_id"] == root.visit_id)
                & (edits_df["t_enter"] >= root.t_enter)
                & (edits_df["t_leave"] <= root.t_leave)
            )
            | (
                bool(root.other_visit_id)
                & ~edits_df["other_t_enter"].isna()
                & (edits_df["other_visit_id"] == root.other_visit_id)
                & (edits_df["other_t_enter"] >= root.other_t_enter)
                & (edits_df["other_t_leave"] <= root.other_t_leave)
            ),
            "subtree_index",
        ] = i

        i += 1

    edits_df["id"] = edits_df["id"].apply(standarize_id)
    edits_df["parent_id"] = edits_df["parent_id"].apply(standarize_id)
    edits_df["other_id"] = edits_df["other_id"].apply(standarize_id)
    edits_df["other_parent_id"] = edits_df["other_parent_id"].apply(standarize_id)

    return edits_df


def prepare_visit_pairs(
    features_path: Path,
    crawl_dir: Path,
):
    features = pd.read_csv(features_path, float_precision="round_trip")

    db_file: Path = crawl_dir / "crawl-data.sqlite"
    exp_file: Path = crawl_dir / "experiments.csv"

    with Database(db_file, exp_file) as database:
        sites_visits = database.sites_visits()

    pair_labels = {
        ("no_fixed", False): ["visit_id_u", "visit_id_a"],
        ("fixed_broken", False): ["visit_id_a", "visit_id_b"],
        ("no_broken", False): ["visit_id_u", "visit_id_b"],
        ("fixed_broken", True): ["visit_id_a", "visit_id_b"],
    }

    features["visit_id_prev"] = 0
    features["visit_id_new"] = 0

    def get_visit_ids(issue_id, visit_labels):
        issue = sites_visits[sites_visits.issue_id.astype(str) == str(issue_id)].iloc[0]
        return issue[visit_labels[0]], issue[visit_labels[1]]

    features[["visit_id_prev", "visit_id_new"]] = features.apply(
        lambda x: get_visit_ids(x["issue"], pair_labels[(x["origin"], x["flipped"])]),
        axis=1,
        result_type="expand",
    )

    return features


def debug(
    crawl_dir: Path,
    issue: int,
    visit_in_label: str,
    visit_out_label: str,
):
    db_file: Path = crawl_dir / "crawl-data.sqlite"
    exp_file: Path = crawl_dir / "experiments.csv"

    with Database(db_file, exp_file) as database:
        sites_visits = database.sites_visits()

        site_visit = sites_visits[sites_visits.issue_id.astype(str) == str(issue)].iloc[
            0
        ]

        _, graphs_in = build_graph(
            database, site_visit[visit_in_label], site_visit.site_url
        )
        visualize_tree(graphs_in[0])

        salient_nodes = graphs_in[0][graphs_in[0].saliency == 1.0]

        _, graphs_out = build_graph(
            database, site_visit[visit_out_label], site_visit.site_url, salient_nodes
        )
        visualize_tree(graphs_out[0])

        display(salient_nodes)

        t, (
            *_,
            salient_nodes,
            common,
            diff,
            nodes,
            edges,
        ) = process_differential_experiment(
            (site_visit[visit_in_label], graphs_in),
            (site_visit[visit_out_label], graphs_out),
            site_visit.browser_id_u,
        )

        # visualize_tree(common)
        _index_subtrees_debug(diff)

        # print(len(diff))

        for _, root in diff[diff.is_root == True].iterrows():
            subtree = get_subtree(diff, root)

            visualize_edit_tree(subtree)
