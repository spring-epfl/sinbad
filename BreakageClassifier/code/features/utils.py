from pathlib import Path
from typing import List
import pandas as pd
import json
import networkx as nx
import re
from openwpm_utils import domain as du
from six.moves.urllib.parse import urlparse, parse_qs
from sklearn import preprocessing
from tqdm import tqdm
from yaml import load, dump
import numpy as np
import traceback

from BreakageClassifier.code.graph.constants import COMMENT_TYPE, DOM_TYPE, REQUEST_TYPE

from .. import graph as gs
import base64
import hashlib

from ..logger import LOGGER

import time


def time_it(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        LOGGER.info(f"{func.__name__} took {end - start:.2f} seconds")
        return end - start, result

    return wrapper


def get_subtree(df: pd.DataFrame, root: pd.Series) -> pd.DataFrame:
    # def _fix_id(_id):
    #     try:
    #         return str(int(_id))
    #     except:
    #         try:
    #             return str(int(float(_id)))
    #         except:
    #             return _id

    # def _fix_ids(row):
    #     row.new_id = _fix_id(row.new_id)
    #     row.new_parent = _fix_id(row.new_parent)
    #     row.prev_id = _fix_id(row.prev_id)
    #     row.prev_parent = _fix_id(row.prev_parent)

    #     return row

    if "subtree_index" not in df.columns:
        raise ValueError(
            "subtree_index column not found in dataframe. run index_subtrees first on the dataframe"
        )

    return (
        df[df["subtree_index"] == root.subtree_index]
        # .apply(_fix_ids, axis=1)
        .fillna(
            {
                "prev_id": -1,
                "new_id": -1,
                "prev_parent": -1,
                "new_parent": -1,
            }
        ).drop_duplicates()
    )


def has_ad_keyword(node, G):
    """
    Function to check if a node URL xhas an ad keyword.

    Args:
      node: URL of node
      G: networkX representation of graph
    Returns:
      has_ad_keyword: binary value showing if node URL has ad keyword
    """

    keyword_raw = [
        "ad",
        "ads",
        "advert",
        "popup",
        "banner",
        "sponsor",
        "iframe",
        "googlead",
        "adsys",
        "adser",
        "advertise",
        "redirect",
        "popunder",
        "punder",
        "popout",
        "click",
        "track",
        "play",
        "pop",
        "prebid",
        "bid",
        "pb.min",
        "affiliate",
        "ban",
        "delivery",
        "promo",
        "tag",
        "zoneid",
        "siteid",
        "pageid",
        "size",
        "viewid",
        "zone_id",
        "google_afc",
        "google_afs",
    ]
    has_ad_keyword = 0
    node_type = G.nodes[node]["type"]
    if node_type != "Element" and node_type != "Storage":
        for key in keyword_raw:
            key_matches = [m.start() for m in re.finditer(key, node, re.I)]
            for key_match in key_matches:
                has_ad_keyword = 1
                break
            if has_ad_keyword == 1:
                break
    return has_ad_keyword


def ad_keyword_ascendants(node, G):
    """
    Function to check if any ascendant of a node has an ad keyword.

    Args:
      node: URL of node
      G: networkX representation of graph
    Returns:
      ascendant_has_ad_keyword: binary value showing if ascendant has ad keyword
    """

    keyword_raw = [
        "ad",
        "ads",
        "advert",
        "popup",
        "banner",
        "sponsor",
        "iframe",
        "googlead",
        "adsys",
        "adser",
        "advertise",
        "redirect",
        "popunder",
        "punder",
        "popout",
        "click",
        "track",
        "play",
        "pop",
        "prebid",
        "bid",
        "pb.min",
        "affiliate",
        "ban",
        "delivery",
        "promo",
        "tag",
        "zoneid",
        "siteid",
        "pageid",
        "size",
        "viewid",
        "zone_id",
        "google_afc",
        "google_afs",
    ]

    ascendant_has_ad_keyword = 0
    ascendants = nx.ancestors(G, node)
    for ascendant in ascendants:
        try:
            node_type = nx.get_node_attributes(G, "type")[ascendant]
            if node_type != "Element" and node_type != "Storage":
                for key in keyword_raw:
                    key_matches = [m.start() for m in re.finditer(key, ascendant, re.I)]
                    for key_match in key_matches:
                        ascendant_has_ad_keyword = 1
                        break
                    if ascendant_has_ad_keyword == 1:
                        break
            if ascendant_has_ad_keyword == 1:
                break
        except:
            continue
    return ascendant_has_ad_keyword


def find_modified_storage(df_target):
    """
    Function to find modified edges -- if a storage element is set by Node 1 and modified (set again) by Node 2,
    there will be an edge from Node 1 to Node 2.

    Args:
      df_target: DataFrame representation of all storage sets, for a particular storage element.
    Returns:
      df_modedges: DataFrame representation of modified edges.
    """

    df_modedges = pd.DataFrame()
    df_copy = df_target.copy().sort_values(by=["time_stamp"])
    df_copy = df_copy.reset_index()
    set_node = df_copy.iloc[[0]][["src", "dst"]]
    modify_nodes = df_copy.drop([0], axis=0)[["src", "dst"]]

    if len(modify_nodes) > 0:
        df_merged = pd.merge(set_node, modify_nodes, on="dst")
        df_modedges = df_merged[["src_x", "src_y", "dst"]].drop_duplicates()
        df_modedges.columns = ["src", "dst", "attr"]
        df_modedges = (
            df_modedges.groupby(["src", "dst"])["attr"].apply(len).reset_index()
        )

    return df_modedges


def get_cookieval(attr):
    """
    Function to extract cookie value.

    Args:
      attr: attributes of cookie node
    Returns:
      name: cookie value
    """

    try:
        attr = json.loads(attr)
        if "value" in attr:
            return attr["value"]
        else:
            return None
    except:
        return None


def get_cookiename(attr):
    """
    Function to extract cookie name.

    Args:
      attr: attributes of cookie node
    Returns:
      name: cookie name
    """

    try:
        attr = json.loads(attr)
        if "name" in attr:
            return attr["name"]
        else:
            return None
    except:
        return None


def get_redirect_depths(df_graph):
    """
    Function to extract redirect depths of every node in the graph.

    Args:
      df_graph: DataFrame representation of graph
    Returns:
      dict_redict: dictionary of redirect depths for each node
    """

    dict_redirect = {}

    try:
        http_status = [300, 301, 302, 303, 307, 308]
        http_status = http_status + [str(x) for x in http_status]
        df_redirect = df_graph[df_graph["response_status"].isin(http_status)]

        G_red = gs.build_networkx_graph(df_redirect)

        for n in G_red.nodes():
            dict_redirect[n] = 0
            dfs_edges = list(nx.edge_dfs(G_red, source=n, orientation="reverse"))
            ct = 0
            depths = []
            if len(dfs_edges) == 1:
                dict_redirect[n] = 1
            if len(dfs_edges) >= 2:
                ct += 1
                for i in range(1, len(dfs_edges)):
                    if dfs_edges[i][1] != dfs_edges[i - 1][0]:
                        depths.append(ct)
                        ct = 1
                    else:
                        ct += 1
                depths.append(ct)
                if len(depths) > 0:
                    dict_redirect[n] = max(depths)

        return dict_redirect

    except Exception as e:
        return dict_redirect


def find_urls(df):
    """
    Function to get set of URLs on a site.

    Args:
      df: DataFrame representation of all edges.
    Returns:
      all_urls: List of URLs.
    """

    src_urls = df["src"].tolist()
    dst_urls = df["dst"].tolist()
    all_urls = list(set(src_urls + dst_urls))
    return all_urls


def check_full_cookie(cookie_value, dest):
    """
    Function to check if a cookie value exists in a URL.

    Args:
      cookie_value: Cookie value
      dest: URL
    Returns:
      Binary value showing whether a cookie value exists in a URL.
    """

    return (
        True
        if len([item for item in cookie_value if item in dest and len(item) > 3]) > 0
        else False
    )


def check_partial_cookie(cookie_value, dest):
    """
    Function to check if a partial cookie value exists in a URL.

    Args:
      cookie_value: Cookie value
      dest: URL
    Returns:
      Binary value showing whether a partial cookie value exists in a URL.
    """

    for value in cookie_value:
        split_cookie = re.split(
            r"\.+|;+|]+|\!+|\@+|\#+|\$+|\%+|\^+|\&+|\*+|\(+|\)+|\-+|\_+|\++|\~+|\`+|\@+=|\{+|\}+|\[+|\]+|\\+|\|+|\:+|\"+|\'+|\<+|\>+|\,+|\?+|\/+",
            value,
        )
        return (
            True
            if len([item for item in split_cookie if item in dest and len(item) > 3])
            > 0
            else False
        )
    return False


def check_base64_cookie(cookie_value, dest):
    """
    Function to check if a base64 encoded cookie value exists in a URL.

    Args:
      cookie_value: Cookie value
      dest: URL
    Returns:
      Binary value showing whether a base64 encoded cookie value exists in a URL.
    """

    return (
        True
        if len(
            [
                item
                for item in cookie_value
                if base64.b64encode(item.encode("utf-8")).decode("utf8") in dest
                and len(item) > 3
            ]
        )
        > 0
        else False
    )


def check_md5_cookie(cookie_value, dest):
    """
    Function to check if a MD5 hashed cookie value exists in a URL.

    Args:
      cookie_value: Cookie value
      dest: URL
    Returns:
      Binary value showing whether a MD5 hashed cookie value exists in a URL.
    """

    return (
        True
        if len(
            [
                item
                for item in cookie_value
                if hashlib.md5(item.encode("utf-8")).hexdigest() in dest
                and len(item) > 3
            ]
        )
        > 0
        else False
    )


def check_sha1_cookie(cookie_value, dest):
    """
    Function to check if a SHA1 hashed cookie value exists in a URL.

    Args:
      cookie_value: Cookie value
      dest: URL
    Returns:
      Binary value showing whether a SHA1 hashed cookie value exists in a URL.
    """

    return (
        True
        if len(
            [
                item
                for item in cookie_value
                if hashlib.sha1(item.encode("utf-8")).hexdigest() in dest
                and len(item) > 3
            ]
        )
        > 0
        else False
    )


def check_full_cookie_set(cookie_value, dest):
    """
    Function to check if a cookie value exists in a URL.

    Args:
      cookie_value: Cookie value
      dest: URL
    Returns:
      Binary value showing whether a cookie value exists in a URL.
    """

    if (len(cookie_value) > 3) and (cookie_value in dest):
        return True
    else:
        return False


def check_partial_cookie_set(cookie_value, dest):
    """
    Function to check if a partial cookie value exists in a URL.

    Args:
      cookie_value: Cookie value
      dest: URL
    Returns:
      Binary value showing whether a partial cookie value exists in a URL.
    """

    split_cookie = re.split(
        r"\.+|;+|]+|\!+|\@+|\#+|\$+|\%+|\^+|\&+|\*+|\(+|\)+|\-+|\_+|\++|\~+|\`+|\@+=|\{+|\}+|\[+|\]+|\\+|\|+|\:+|\"+|\'+|\<+|\>+|\,+|\?+|\/+",
        cookie_value,
    )
    for item in split_cookie:
        if len(item) > 3 and item in dest:
            return True
    return False


def check_base64_cookie_set(cookie_value, dest):
    """
    Function to check if a base64 encoded cookie value exists in a URL.

    Args:
      cookie_value: Cookie value
      dest: URL
    Returns:
      Binary value showing whether a base64 encoded cookie value exists in a URL.
    """

    if (len(cookie_value) > 3) and (
        base64.b64encode(cookie_value.encode("utf-8")).decode("utf8") in dest
    ):
        return True
    else:
        return False


def check_md5_cookie_set(cookie_value, dest):
    """
    Function to check if a MD5 hashed cookie value exists in a URL.

    Args:
      cookie_value: Cookie value
      dest: URL
    Returns:
      Binary value showing whether a MD5 hashed cookie value exists in a URL.
    """

    if (len(cookie_value) > 3) and (
        hashlib.md5(cookie_value.encode("utf-8")).hexdigest() in dest
    ):
        return True
    else:
        return False


def check_sha1_cookie_set(cookie_value, dest):
    """
    Function to check if a SHA1 hashed cookie value exists in a URL.

    Args:
      cookie_value: Cookie value
      dest: URL
    Returns:
      Binary value showing whether a SHA1 hashed cookie value exists in a URL.
    """

    if (len(cookie_value) > 3) and (
        hashlib.sha1(cookie_value.encode("utf-8")).hexdigest() in dest
    ):
        return True
    else:
        return False


def check_cookie_presence(http_attr, dest):
    check_value = False

    try:
        http_attr = json.loads(http_attr)

        for item in http_attr:
            if "Cookie" in item[0]:
                cookie_pairs = item[1].split(";")
                for cookie_pair in cookie_pairs:
                    cookie_value = cookie_pair.strip().split("=")[1:]
                    full_cookie = check_full_cookie(cookie_value, dest)
                    partial_cookie = check_partial_cookie(cookie_value, dest)
                    base64_cookie = check_base64_cookie(cookie_value, dest)
                    md5_cookie = check_md5_cookie(cookie_value, dest)
                    sha1_cookie = check_sha1_cookie(cookie_value, dest)
                    check_value = (
                        check_value
                        | full_cookie
                        | partial_cookie
                        | base64_cookie
                        | md5_cookie
                        | sha1_cookie
                    )
                    if check_value:
                        return check_value
    except:
        check_value = False
    return check_value


def find_indirect_edges(G, df_graph):
    """
    Function to extract shared information edges, used for dataflow features.

    Args:
      G: networkX graph
      df_graph: DataFrame representation of graph
    Returns:
      df_edges: DataFrame representation of shared information edges.
    """

    df_edges = pd.DataFrame()

    try:
        storage_set = df_graph[
            (df_graph["action"] == "set")
            | (df_graph["action"] == "set_js")
            | (df_graph["action"] == "set_storage_js")
        ][["src", "dst"]]
        storage_get = df_graph[
            (df_graph["action"] == "get")
            | (df_graph["action"] == "get_js")
            | (df_graph["action"] == "get_storage_js")
        ][["src", "dst"]]

        # Nodes that set to nodes that get
        df_merged = pd.merge(storage_set, storage_get, on="dst")
        df_get_edges = df_merged[["src_x", "src_y", "dst"]].drop_duplicates()
        if len(df_get_edges) > 0:
            df_get_edges.columns = ["src", "dst", "attr"]
            df_get_edges["cookie"] = df_get_edges["attr"]
            df_get_edges = (
                df_get_edges.groupby(["src", "dst"])["attr"].apply(len).reset_index()
            )
            df_get_edges["type"] = "set_get"
            df_edges = pd.concat([df_edges, df_get_edges], ignore_index=True)

        # Nodes that set to nodes that modify
        all_storage_set = df_graph[
            (df_graph["action"] == "set")
            | (df_graph["action"] == "set_js")
            | (df_graph["action"] == "set_storage_js")
            | (df_graph["action"] == "remove_storage_js")
        ]
        df_modified_edges = all_storage_set.groupby("dst").apply(find_modified_storage)
        if len(df_modified_edges) > 0:
            df_modified_edges["type"] = "set_modify"
            df_edges = pd.concat([df_edges, df_modified_edges], ignore_index=True)

        # Nodes that set to URLs with cookie value
        df_set_url_edges = pd.DataFrame()
        df_cookie_set = df_graph[
            (df_graph["action"] == "set") | (df_graph["action"] == "set_js")
        ].copy()
        df_cookie_set["cookie_val"] = df_cookie_set["attr"].apply(get_cookieval)
        cookie_values = list(
            set(
                df_cookie_set[~df_cookie_set["cookie_val"].isnull()][
                    "cookie_val"
                ].tolist()
            )
        )

        df_nodes = df_graph[
            (df_graph["graph_attr"] == "Node")
            & (
                (df_graph["type"] == "Request")
                | (df_graph["type"] == "Script")
                | (df_graph["type"] == "Document")
            )
        ]["name"]
        urls = df_nodes.tolist()
        check_set_value = False

        for dest in urls:
            for cookie_value in cookie_values:
                full_cookie = check_full_cookie_set(cookie_value, dest)
                partial_cookie = check_partial_cookie_set(cookie_value, dest)
                base64_cookie = check_base64_cookie_set(cookie_value, dest)
                md5_cookie = check_md5_cookie_set(cookie_value, dest)
                sha1_cookie = check_sha1_cookie_set(cookie_value, dest)
                check_set_value = (
                    full_cookie
                    | partial_cookie
                    | base64_cookie
                    | md5_cookie
                    | sha1_cookie
                )
                if check_set_value:
                    src = df_cookie_set[df_cookie_set["cookie_val"] == cookie_value][
                        "src"
                    ].iloc[0]
                    dst = dest
                    attr = 1
                    df_set_url_edges = pd.concat(
                        [
                            df_set_url_edges,
                            pd.DataFrame.from_records(
                                [{"src": src, "dst": dst, "attr": attr}]
                            ),
                        ],
                        ignore_index=True,
                    )

        if len(df_set_url_edges) > 0:
            df_set_url_edges = (
                df_set_url_edges.groupby(["src", "dst"])["attr"]
                .apply(len)
                .reset_index()
            )
            df_set_url_edges["type"] = "set_url"
            df_edges = pd.concat([df_edges, df_set_url_edges], ignore_index=True)

        # Nodes that get to URLs with cookie value
        df_http_requests = df_graph[
            (df_graph["reqattr"] != "CS")
            & (df_graph["src"] != "N/A")
            & (df_graph["action"] != "CS")
            & (df_graph["graph_attr"] != "EdgeWG")
        ]
        df_http_requests_merge = pd.merge(
            left=df_http_requests,
            right=df_http_requests,
            how="inner",
            left_on=["visit_id", "dst"],
            right_on=["visit_id", "src"],
        )
        df_http_requests_merge = df_http_requests_merge[
            df_http_requests_merge["reqattr_x"].notnull()
        ]

        if len(df_http_requests_merge):
            df_http_requests_merge["cookie_presence"] = df_http_requests_merge.apply(
                axis=1, func=lambda x: check_cookie_presence(x["reqattr_x"], x["dst_y"])
            )

            df_get_url_edges = df_http_requests_merge[
                df_http_requests_merge["cookie_presence"] == True
            ][["src_x", "dst_y", "attr_x"]]
            if len(df_get_url_edges) > 0:
                df_get_url_edges.columns = ["src", "dst", "attr"]
                df_get_url_edges = (
                    df_get_url_edges.groupby(["src", "dst"])["attr"]
                    .apply(len)
                    .reset_index()
                )
                df_get_url_edges["type"] = "get_url"
                df_edges = pd.concat([df_edges, df_get_url_edges], ignore_index=True)

    except Exception as e:
        LOGGER.exception("An error occurred when extracting shared information edges.")
        return df_edges

    return df_edges


def add_missing_cols(df, cols):
    for col in cols:
        if col not in df.columns:
            df[col] = None

    return df


def standarize_id(x):
    try:
        return int(x)
    except:
        try:
            return int(float(x))
        except:
            return str(x)


@time_it
def index_subtrees(edits_df):
    LOGGER.debug("Indexing subtrees")
    edits_df["subtree_index"] = -1
    roots = edits_df[edits_df.is_root]

    i = 0

    for _, root in tqdm(roots.iterrows(), total=len(roots)):
        edits_df.loc[
            (edits_df["prev_vid"] == root.prev_vid)
            & (edits_df["new_vid"] == root.new_vid)
            & (
                (
                    ~edits_df["prev_te"].isna()
                    & (edits_df["prev_te"] >= root.prev_te)
                    & (edits_df["prev_tl"] <= root.prev_tl)
                )
                | (
                    ~edits_df["new_te"].isna()
                    & (edits_df["new_te"] >= root.new_te)
                    & (edits_df["new_tl"] <= root.new_tl)
                )
            ),
            "subtree_index",
        ] = i

        i += 1

    edits_df["prev_id"] = edits_df["prev_id"].apply(standarize_id)
    edits_df["prev_parent"] = edits_df["prev_parent"].apply(standarize_id)
    edits_df["new_id"] = edits_df["new_id"].apply(standarize_id)
    edits_df["new_parent"] = edits_df["new_parent"].apply(standarize_id)
    return edits_df


@time_it
def mix_features(*tree_features: pd.DataFrame) -> pd.DataFrame:
    """Given features dataframes that share a set of sample identifying columns, it returns a dataframe containing
    all columns without any duplication.

    ex: if features1.columns = [id, vid, n] and features2.columns = [id, vid, m] it returns features s.t.
        features.columns = [id, vid, n, m]

    Returns:
        DataFrame: the resulting features table.
    """

    _tree_features = tree_features[0]

    _tree_features_cols = set(_tree_features.columns.tolist())

    if len(tree_features) > 1:
        for i in range(1, len(tree_features)):
            _next_tree_features = tree_features[i]
            _cols = set(_next_tree_features.columns.tolist())
            _cols_to_add = list(_cols - _tree_features_cols)

            _tree_features = pd.concat(
                [_tree_features, _next_tree_features[_cols_to_add]], axis=1
            )

    return _tree_features


def _parse_attrs(row: pd.Series):
    
    
    if "attributes" in row and isinstance(row.attributes, str):
        try:
            attrs = json.loads(row.attributes)
        except:
            raise Exception(f"Not json: {row.attributes}")
        
    elif "other_attributes" in row and isinstance(row.other_attributes, str):
        try:
            attrs = json.loads(row.other_attributes)
        except:
            raise Exception(f"Not json: {row.other_attributes}")
    elif "prev_attr" in row and isinstance(row.prev_attr, str):
        try:
            attrs = json.loads(row.prev_attr)
        except:
            raise Exception(f"Not json: {row.prev_attr}")
        
    elif "new_attr" in row and isinstance(row.new_attr, str):
        try:
            attrs = json.loads(row.new_attr)
        except:
            raise Exception(f"Not json: {row.new_attr}")
    else:
        print(row)
        return {}
    
    return {x["key"]: x["value"] for x in attrs}


def _node_name(node):
    if node.type == REQUEST_TYPE:
        return node.nodeName + ": " + _parse_attrs(node)["url"]

    if node.saliency:
        return node.nodeName + " [SALIENT]"

    return node.nodeName


def _node_summary(node, labeled=False):
    if labeled:
        node["nodeName"] = node.tag
        node["rem_in_l"] = node.added
        node["rem_in_r"] = node.removed
        node['attr_class'] = node.prev_attr_class if node.added else node.new_attr_class

    node_name = _node_name(node)

    suffix = ""
    
    if isinstance(node.attr_class, str):
        attr_str = ".".join(node.attr_class.split(" "))
        suffix += f" .{attr_str}"

    if node.rem_in_l:
        suffix += "(+)"

    elif node.rem_in_r:
        suffix += "(-)"

    elif node.edited:
        suffix += "(edited)"

    # if node.type == COMMENT_TYPE:
    #     suffix += " " + json.loads(node.visual_cues)["text"]

    return node_name + " " + suffix


def visualize_tree(tree_df: pd.DataFrame, root_id=-1):
    from treelib import Node, Tree

    tree = Tree()

    for _, node in tree_df.iterrows():
        if node.parent_id == root_id:
            tree.create_node(f"[{node.id}] {_node_name(node)}", node.id)
        else:
            tree.create_node(
                f"[{node.id}] {_node_name(node)}",
                node.id,
                parent=node.parent_id,
            )

    print(tree.show(stdout=False))


def visualize_edit_tree(tree_df: pd.DataFrame, labeled=False):
    # display(tree_df[["prev_vid", "new_vid", "origin", "prev_id", "new_id", "is_root", "added", "removed"]])
    from treelib import Node, Tree

    if labeled:
        cols = {
            "parent_id": "prev_parent",
            "other_parent_id": "new_parent",
            "id": "prev_id",
            "other_id": "new_id",
        }
    else:
        cols = {
            "parent_id": "parent_id",
            "other_parent_id": "parent_id",
            "id": "id",
            "other_id": "other_id",
        }

    tree = Tree()

    def _node_id(node):
        return f"{node[cols['id']]}={node[cols['other_id']]}"

    def _node_parent_id(node):
        return f"{node[cols['parent_id']]}={node[cols['other_parent_id']]}"

    root = None

    for _, node in tree_df.iterrows():
        if node.is_root:
            root = node
            tree.create_node(
                f"[{_node_id(node)}] {_node_summary(node, labeled)}", _node_id(node)
            )
        else:
            parent = _node_parent_id(node)
            if not tree.contains(parent):
                parent = _node_id(root)

            tree.create_node(
                f"[{_node_id(node)}] {_node_summary(node, labeled)}",
                _node_id(node),
                parent=parent,
            )

    print(tree.show(stdout=False))
