import pandas as pd
import networkx as nx

from BreakageClassifier.code.graph.constants import (
    ERROR_TYPE,
    INTERACT_ERROR_TYPE,
    INTERACT_RELATE_DOM_TYPE,
    INTERACTION_TYPE,
    SCRIPT_RELATE_DOM_TYPE,
    SCRIPT_TYPE,
)
from .utils import *

from ..logger import LOGGER
from ..utils import graph_node_id


def get_storage_features(df_graph, node):
    """
    Function to extract storage features.

    Args:
      df_graph: DataFrame representation of graph
      node: URL of node
    Returns:
      storage_features: storage feature values
      storage_feature_names: storage feature names
    """

    cookie_get = df_graph[
        (df_graph["src"] == node)
        & ((df_graph["action"] == "get") | (df_graph["action"] == "get_js"))
    ]

    cookie_set = df_graph[
        (df_graph["src"] == node)
        & ((df_graph["action"] == "set") | (df_graph["action"] == "set_js"))
    ]

    localstorage_get = df_graph[
        (df_graph["src"] == node) & (df_graph["action"] == "get_storage_js")
    ]

    localstorage_set = df_graph[
        (df_graph["src"] == node) & (df_graph["action"] == "set_storage_js")
    ]

    num_get_storage = len(cookie_get) + len(localstorage_get)
    num_set_storage = len(cookie_set) + len(localstorage_set)
    num_get_cookie = len(cookie_get)
    num_set_cookie = len(cookie_set)

    storage_features = [
        num_get_storage,
        num_set_storage,
        num_get_cookie,
        num_set_cookie,
    ]
    storage_feature_names = [
        "num_get_storage",
        "num_set_storage",
        "num_get_cookie",
        "num_set_cookie",
    ]

    return storage_features, storage_feature_names


def get_redirect_features(df_graph, node, dict_redirect):
    """
    Function to extract redirect features.

    Args:
      df_graph: DataFrame representation of graph
      node: URL of node
      dict_redirect: dictionary of redirect depths for each node
    Returns:
      redirect_features: redirect feature values
      redirect_feature_names: redirect feature names
    """

    http_status = [300, 301, 302, 303, 307, 308]
    http_status = http_status + [str(x) for x in http_status]

    redirects_sent = df_graph[
        (df_graph["src"] == node) & (df_graph["response_status"].isin(http_status))
    ]
    redirects_rec = df_graph[
        (df_graph["dst"] == node) & (df_graph["response_status"].isin(http_status))
    ]
    num_redirects_sent = len(redirects_sent)
    num_redirects_rec = len(redirects_rec)

    max_depth_redirect = 0
    if node in dict_redirect:
        max_depth_redirect = dict_redirect[node]

    redirect_features = [num_redirects_sent, num_redirects_rec, max_depth_redirect]
    redirect_feature_names = [
        "num_redirects_sent",
        "num_redirects_rec",
        "max_depth_redirect",
    ]

    return redirect_features, redirect_feature_names


def get_request_flow_features(G, df_graph, node):
    """
    Function to extract request flow features.

    Args:
      G: networkX graph
      df_graph: DataFrame representation of graph
      node: URL of node
    Returns:
      rf_features: request flow feature values
      rf_feature_names: request flow feature names
    """

    requests_sent = df_graph[
        (df_graph["src"] == node)
        & (df_graph["reqattr"].notnull())
        & (df_graph["reqattr"] != "CS")
        & (df_graph["reqattr"] != "N/A")
    ]
    requests_received = df_graph[
        (df_graph["dst"] == node)
        & (df_graph["reqattr"].notnull())
        & (df_graph["reqattr"] != "CS")
        & (df_graph["reqattr"] != "N/A")
    ]
    num_requests_sent = len(requests_sent)
    num_requests_received = len(requests_received)

    # Request flow features
    predecessors = list(G.predecessors(node))
    successors = list(G.successors(node))
    predecessors_type = [G.nodes[x].get("type") for x in predecessors]
    num_script_predecessors = len([x for x in predecessors_type if x == "Script"])
    successors_type = [G.nodes[x].get("type") for x in successors]
    num_script_successors = len([x for x in successors_type if x == "Script"])

    rf_features = [
        num_script_predecessors,
        num_script_successors,
        num_requests_sent,
        num_requests_received,
    ]

    rf_feature_names = [
        "num_script_predecessors",
        "num_script_successors",
        "num_requests_sent",
        "num_requests_received",
    ]

    return rf_features, rf_feature_names


def get_indirect_features(G, df_graph, node):
    """
    Function to extract indirect edge features.

    Args:
      G: networkX graph of indirect edges
      df_graph: DataFrame representation of graph (indirect edges only)
      node: URL of node
    Returns:
      indirect_features: indirect feature values
      indirect_feature_names: indirect feature names
    """

    in_degree = -1
    out_degree = -1
    ancestors = -1
    descendants = -1
    closeness_centrality = -1
    average_degree_connectivity = -1
    eccentricity = -1
    mean_in_weights = -1
    min_in_weights = -1
    max_in_weights = -1
    mean_out_weights = -1
    min_out_weights = -1
    max_out_weights = -1
    num_set_get_src = 0
    num_set_mod_src = 0
    num_set_url_src = 0
    num_get_url_src = 0
    num_set_get_dst = 0
    num_set_mod_dst = 0
    num_set_url_dst = 0
    num_get_url_dst = 0

    try:
        if len(df_graph) > 0:
            num_set_get_src = len(
                df_graph[(df_graph["type"] == "set_get") & (df_graph["src"] == node)]
            )
            num_set_mod_src = len(
                df_graph[(df_graph["type"] == "set_modify") & (df_graph["src"] == node)]
            )
            num_set_url_src = len(
                df_graph[(df_graph["type"] == "set_url") & (df_graph["src"] == node)]
            )
            num_get_url_src = len(
                df_graph[(df_graph["type"] == "get_url") & (df_graph["src"] == node)]
            )
            num_set_get_dst = len(
                df_graph[(df_graph["type"] == "set_get") & (df_graph["dst"] == node)]
            )
            num_set_mod_dst = len(
                df_graph[(df_graph["type"] == "set_modify") & (df_graph["dst"] == node)]
            )
            num_set_url_dst = len(
                df_graph[(df_graph["type"] == "set_url") & (df_graph["dst"] == node)]
            )
            num_get_url_dst = len(
                df_graph[(df_graph["type"] == "get_url") & (df_graph["dst"] == node)]
            )

        if (len(G.nodes()) > 0) and (node in G.nodes()):
            in_degree = G.in_degree(node)
            out_degree = G.in_degree(node)
            ancestors = len(nx.ancestors(G, node))
            descendants = len(nx.descendants(G, node))
            closeness_centrality = nx.closeness_centrality(G, node)
            average_degree_connectivity = [
                *nx.average_degree_connectivity(G, nodes=[node]).values()
            ][0]
            try:
                H = G.copy().to_undirected()
                eccentricity = nx.eccentricity(H, node)
            except Exception as e:
                eccentricity = -1
            in_weights = df_graph[(df_graph["dst"] == node)]["attr"].tolist()
            out_weights = df_graph[(df_graph["src"] == node)]["attr"].tolist()

            if len(in_weights) > 0:
                mean_in_weights = np.mean(in_weights)
                min_in_weights = min(in_weights)
                max_in_weights = max(in_weights)

            if len(out_weights) > 0:
                mean_out_weights = np.mean(out_weights)
                min_out_weights = min(out_weights)
                max_out_weights = max(out_weights)
    except Exception as e:
        LOGGER.warning("[ get_indirect_features ] : ERROR - ", exc_info=True)

    indirect_features = [
        in_degree,
        out_degree,
        ancestors,
        descendants,
        closeness_centrality,
        average_degree_connectivity,
        eccentricity,
        mean_in_weights,
        min_in_weights,
        max_in_weights,
        mean_out_weights,
        min_out_weights,
        max_out_weights,
        num_set_get_src,
        num_set_mod_src,
        num_set_url_src,
        num_get_url_src,
        num_set_get_dst,
        num_set_mod_dst,
        num_set_url_dst,
        num_get_url_dst,
    ]

    indirect_feature_names = [
        "indirect_in_degree",
        "indirect_out_degree",
        "indirect_ancestors",
        "indirect_descendants",
        "indirect_closeness_centrality",
        "indirect_average_degree_connectivity",
        "indirect_eccentricity",
        "indirect_mean_in_weights",
        "indirect_min_in_weights",
        "indirect_max_in_weights",
        "indirect_mean_out_weights",
        "indirect_min_out_weights",
        "indirect_max_out_weights",
        "num_set_get_src",
        "num_set_mod_src",
        "num_set_url_src",
        "num_get_url_src",
        "num_set_get_dst",
        "num_set_mod_dst",
        "num_set_url_dst",
        "num_get_url_dst",
    ]

    return indirect_features, indirect_feature_names


def get_indirect_all_features(G, node):
    """
    Function to extract dataflow features of graph with both direct and indirect edges ('indirect_all').

    Args:
      G: networkX graph (of both direct and indirect edges)
      node: URL of node
    Returns:
      indirect_all_features: indirect_all feature values
      indirect_all_feature_names: indirect_all feature names
    """

    in_degree = -1
    out_degree = -1
    ancestors = -1
    descendants = -1
    closeness_centrality = -1
    average_degree_connectivity = -1
    eccentricity = -1

    try:
        if (len(G.nodes()) > 0) and (node in G.nodes()):
            in_degree = G.in_degree(node)
            out_degree = G.in_degree(node)
            ancestors = len(nx.ancestors(G, node))
            descendants = len(nx.descendants(G, node))
            closeness_centrality = nx.closeness_centrality(G, node)
            average_degree_connectivity = [
                *nx.average_degree_connectivity(G, nodes=[node]).values()
            ][0]
            try:
                H = G.copy().to_undirected()
                eccentricity = nx.eccentricity(H, node)
            except Exception as e:
                eccentricity = -1
    except Exception as e:
        LOGGER.warning("[ get_indirect_all_features ] : ERROR - ", exc_info=True)

    indirect_all_features = [
        in_degree,
        out_degree,
        ancestors,
        descendants,
        closeness_centrality,
        average_degree_connectivity,
        eccentricity,
    ]
    indirect_all_feature_names = [
        "indirect_all_in_degree",
        "indirect_all_out_degree",
        "indirect_all_ancestors",
        "indirect_all_descendants",
        "indirect_all_closeness_centrality",
        "indirect_all_average_degree_connectivity",
        "indirect_all_eccentricity",
    ]

    return indirect_all_features, indirect_all_feature_names


def get_dataflow_features(
    G, df_graph, node, dict_redirect, G_indirect, G_indirect_all, df_indirect_graph
):
    """
    Function to extract dataflow features. This function calls
    the other functions to extract different types of dataflow features.

    Args:
      G: networkX graph
      df_graph: DataFrame representation of graph
      node: URL of node
      dict_redirect: dictionary of redirect depths for each node
      G_indrect: networkX graph of indirect edges
      G_indirect_all: networkX graph of direct and indirect edges
      df_indirect_graph: DataFrame representation of indirect edges
    Returns:
      all_features: dataflow feature values
      all_feature_names: dataflow feature names
    """

    all_features = []
    all_feature_names = []

    storage_features, storage_feature_names = get_storage_features(df_graph, node)
    redirect_features, redirect_feature_names = get_redirect_features(
        df_graph, node, dict_redirect
    )
    rf_features, rf_feature_names = get_request_flow_features(G, df_graph, node)
    indirect_features, indirect_feature_names = get_indirect_features(
        G_indirect, df_indirect_graph, node
    )
    indirect_all_features, indirect_all_feature_names = get_indirect_all_features(
        G_indirect_all, node
    )

    all_features = (
        storage_features
        + redirect_features
        + rf_features
        + indirect_features
        + indirect_all_features
    )
    all_feature_names = (
        storage_feature_names
        + redirect_feature_names
        + rf_feature_names
        + indirect_feature_names
        + indirect_all_feature_names
    )

    return all_features, all_feature_names


def pre_extraction(G, df_graph):
    """
    Function to obtain indirect edges before calculating dataflow features.

    Args:
      G: networkX graph
      df_graph: DataFrame representation of graph
    Returns:
      dict_redirect: dictionary of redirect depths for each node
      G_indrect: networkX graph of indirect edges
      G_indirect_all: networkX graph of direct and indirect edges
      df_indirect_edges: DataFrame representation of indirect edges
    """

    G_indirect = nx.DiGraph()
    dict_redirect = get_redirect_depths(df_graph)
    df_indirect_edges = find_indirect_edges(G, df_graph)

    if len(df_indirect_edges) > 0:
        G_indirect = nx.from_pandas_edgelist(
            df_indirect_edges,
            source="src",
            target="dst",
            edge_attr=True,
            create_using=nx.DiGraph(),
        )
    G_indirect_all = nx.compose(G, G_indirect)

    return dict_redirect, G_indirect, G_indirect_all, df_indirect_edges


def _get_script_features_subtree(
    root: pd.Series,
    df_edits: pd.DataFrame,
    cols: list,
    env_nodes: pd.DataFrame,
    env_edges: pd.DataFrame,
):
    subtree = get_subtree(df_edits, root)

    subtree_nodes = set(
        subtree.apply(
            lambda row: graph_node_id(row.prev_bid, row.prev_vid, row.prev_id), axis=1
        ).values.tolist()
    )
    subtree_nodes |= set(
        subtree.apply(
            lambda row: graph_node_id(row.new_bid, row.new_vid, row.new_id), axis=1
        ).values.tolist()
    )

    _env_edges = env_edges[
        (
            (env_edges.issue == root.issue)
            & (env_edges.origin == root.origin)
            & (env_edges.type == SCRIPT_RELATE_DOM_TYPE)
            & (
                env_edges.right_dst.isin(subtree_nodes)
                | env_edges.left_dst.isin(subtree_nodes)
            )
        )
    ]

    _scripts_touching = set(
        _env_edges[~_env_edges.removed | _env_edges.added].src.unique()
    )

    n_el_in_scr_tree = _env_edges["count"].sum()
    dn_el_in_scr_tree = _env_edges["dcount"].sum()

    n_scr_in_el_tree = len(_env_edges.src.unique())
    dn_scr_in_el_tree = len(
        _env_edges[
            (_env_edges.removed == True) | (_env_edges.added == True)
        ].src.unique()
    )

    # errors
    _env_edges = env_edges[
        (env_edges.issue == root.issue)
        & (env_edges.origin == root.origin)
        & (env_edges.type.isin([INTERACT_ERROR_TYPE, ERROR_TYPE]))
        & env_edges.src.isin(_scripts_touching)
    ]

    dn_err_in_scr_tree = _env_edges["dcount"].sum()

    # requests
    _env_edges = env_edges[
        (env_edges.issue == root.issue)
        & (env_edges.origin == root.origin)
        & (env_edges.type == REQUEST_TYPE)
        & env_edges.src.isin(_scripts_touching)
    ]

    n_req_in_scr_tree = len(_env_edges.src.unique())
    dn_req_in_scr_tree = _env_edges["dcount"].sum()

    return (
        n_el_in_scr_tree,
        dn_el_in_scr_tree,
        n_scr_in_el_tree,
        dn_scr_in_el_tree,
        dn_err_in_scr_tree,
        n_req_in_scr_tree,
        dn_req_in_scr_tree,
    )


def get_script_features(
    features: pd.DataFrame,
    df_edits: pd.DataFrame,
    env_nodes: pd.DataFrame,
    env_edges: pd.DataFrame,
):
    cols = [
        "n_el_in_scr_tree",  # number of elements touched by scripts  (base + increase/decrease)
        "dn_el_in_scr_tree",
        "n_scr_in_el_tree",  # number of scripts touching this subtree (base + increase/decrease)
        "dn_scr_in_el_tree",
        "dn_err_in_scr_tree",  # increase/decrease in number of errors in scripts touching this subtree
        "n_req_in_scr_tree",  # number of requests with response by script touching this subtree
        "dn_req_in_scr_tree",  # increase/decrease in number of requests in scripts touching this subtree
    ]

    features[cols] = features.progress_apply(
        lambda row: _get_script_features_subtree(
            row, df_edits, cols, env_nodes, env_edges
        ),
        axis=1,
        result_type="expand",
    )


def _get_interaction_features_subtree(
    root: pd.Series,
    df_edits: pd.DataFrame,
    cols: list,
    env_nodes: pd.DataFrame,
    env_edges: pd.DataFrame,
):
    subtree = get_subtree(df_edits, root)

    subtree_nodes = set(
        subtree.apply(
            lambda row: graph_node_id(row.prev_bid, row.prev_vid, row.prev_id), axis=1
        ).values.tolist()
    )

    subtree_nodes |= set(
        subtree.apply(
            lambda row: graph_node_id(row.new_bid, row.new_vid, row.new_id), axis=1
        ).values.tolist()
    )

    subtree_interactions = set(
        env_nodes[
            (env_nodes["type"] == INTERACTION_TYPE)
            & (
                (env_nodes["value"].isin(subtree.prev_id.values.tolist()))
                | (env_nodes["value"].isin(subtree.new_id.values.tolist()))
            )
        ]["name"].values.tolist()
    )

    _env_edges = env_edges[
        (
            (env_edges.issue == root.issue)
            & (env_edges.origin == root.origin)
            & (env_edges.type == INTERACT_RELATE_DOM_TYPE)
            & (
                env_edges.right_dst.isin(subtree_nodes)
                | env_edges.left_dst.isin(subtree_nodes)
            )
        )
    ]

    _interact_touching = set(
        _env_edges[~_env_edges.removed | _env_edges.added].src.unique()
    )

    n_int_in_tree = len(_interact_touching)

    dn_el_in_int_tree = _env_edges["dcount"].sum()

    _env_edges = env_edges[
        (env_edges.issue == root.issue)
        & (env_edges.origin == root.origin)
        & (env_edges.type.isin([INTERACT_ERROR_TYPE, ERROR_TYPE]))
        & (env_edges.src.isin(subtree_interactions))
    ]

    dn_err_in_int_tree = len(_env_edges.src.unique())

    return (n_int_in_tree, dn_el_in_int_tree, dn_err_in_int_tree)


def get_interaction_features(
    features: pd.DataFrame,
    df_edits: pd.DataFrame,
    env_nodes: pd.DataFrame,
    env_edges: pd.DataFrame,
):
    cols = [
        "n_int_in_tree",  # number of interactions touching this subtree
        "dn_el_in_int_tree",  # increase/decrease number of elements touched by interactions
        "dn_err_in_int_tree",  # increase/decrease number of errors in interaction from errors in this subtree
    ]

    features[cols] = features.progress_apply(
        lambda row: _get_interaction_features_subtree(
            row, df_edits, cols, env_nodes, env_edges
        ),
        axis=1,
        result_type="expand",
    )


@time_it
def get_dom_interactive_features(
    df_edits: pd.DataFrame, env_nodes: pd.DataFrame, env_edges: pd.DataFrame
):
    features_df = df_edits[df_edits["is_root"] == True].copy(deep=True)

    LOGGER.debug("Interactive Features: Script features")
    get_script_features(features_df, df_edits, env_nodes, env_edges)

    LOGGER.debug("Interactive Features: Interaction features")

    get_interaction_features(features_df, df_edits, env_nodes, env_edges)

    return features_df


@time_it
def get_global_features(
    df_tree_features: pd.DataFrame,
    env_nodes: pd.DataFrame,
    env_edges: pd.DataFrame,
):
    LOGGER.debug("Global Features: computing...")

    df_tree_features["n_tree_rem"] = 0
    df_tree_features.loc[df_tree_features.removed, "n_tree_rem"] = (
        df_tree_features[df_tree_features.removed]
        .groupby(by=["issue", "origin", "flipped"], dropna=False)
        .prev_id.transform("size")
    )

    df_tree_features["n_tree_add"] = 0
    df_tree_features.loc[df_tree_features.added, "n_tree_add"] = (
        df_tree_features[df_tree_features.added]
        .groupby(by=["issue", "origin", "flipped"], dropna=False)
        .prev_id.transform("size")
    )

    df_tree_features["n_tree_edit"] = 0
    df_tree_features.loc[df_tree_features.edited, "n_tree_edit"] = (
        df_tree_features[df_tree_features.edited]
        .groupby(by=["issue", "origin", "flipped"], dropna=False)
        .prev_id.transform("size")
    )

    df_tree_features["n_sal_tree_add"] = 0

    df_tree_features.loc[
        (df_tree_features.saliency == True) & df_tree_features.added, "n_sal _tree_add"
    ] = (
        df_tree_features[(df_tree_features.saliency == True) & df_tree_features.added]
        .groupby(by=["issue", "origin", "flipped"], dropna=False)
        .prev_id.transform("size")
    )

    df_tree_features["n_sal_tree_rem"] = 0
    df_tree_features.loc[
        (df_tree_features.saliency == True) & df_tree_features.removed, "n_sal_tree_rem"
    ] = (
        df_tree_features[(df_tree_features.saliency == True) & df_tree_features.removed]
        .groupby(by=["issue", "origin", "flipped"], dropna=False)
        .prev_id.transform("size")
    )

    df_tree_features["n_sal_tree_edit"] = 0
    df_tree_features.loc[
        (df_tree_features.saliency == True) & df_tree_features.edited, "n_sal_tree_edit"
    ] = (
        df_tree_features[(df_tree_features.saliency == True) & df_tree_features.edited]
        .groupby(by=["issue", "origin", "flipped"], dropna=False)
        .prev_id.transform("size")
    )

    # df_tree_features["n_scr"] = 0

    _scr = (
        env_nodes[
            (env_nodes.type == SCRIPT_TYPE) & ~env_nodes.removed & ~env_nodes.added
        ][["issue", "origin", "flipped", "name"]]
        .drop_duplicates()
        .groupby(by=["issue", "origin", "flipped"], dropna=False)["name"]
        .apply("size")
        .to_frame("n_scr")
        .reset_index()
    )

    _x = df_tree_features.merge(_scr, on=["issue", "origin", "flipped"], how="left")

    df_tree_features["n_scr"] = _x["n_scr"].fillna(0).tolist()

    _src_add = (
        env_nodes[(env_nodes.type == SCRIPT_TYPE) & env_nodes.added][
            ["issue", "origin", "flipped", "name"]
        ]
        .groupby(by=["issue", "origin", "flipped"], dropna=False)["name"]
        .apply("size")
        .to_frame("n_scr_add")
        .reset_index()
    )

    df_tree_features["n_scr_add"] = (
        df_tree_features.merge(_src_add, on=["issue", "origin", "flipped"], how="left")[
            "n_scr_add"
        ]
        .fillna(0)
        .tolist()
    )

    _src_rem = (
        env_nodes[(env_nodes.type == SCRIPT_TYPE) & env_nodes.removed][
            ["issue", "origin", "flipped", "name"]
        ]
        .groupby(by=["issue", "origin", "flipped"], dropna=False)["name"]
        .apply("size")
        .to_frame("n_scr_rem")
        .reset_index()
    )

    df_tree_features["n_scr_rem"] = (
        df_tree_features.merge(_src_rem, on=["issue", "origin", "flipped"], how="left")[
            "n_scr_rem"
        ]
        .fillna(0)
        .tolist()
    )

    _err_edges = env_edges[(env_edges.type.isin([INTERACT_ERROR_TYPE, ERROR_TYPE]))]

    _err_edges = set(_err_edges.src.unique().tolist()) | set(
        _err_edges.dst.unique().tolist()
    )

    err_scr_nodes = env_nodes[
        (env_nodes.type == SCRIPT_TYPE) & env_nodes.name.isin(_err_edges)
    ]

    _scr_err = (
        err_scr_nodes[
            (err_scr_nodes.type == SCRIPT_TYPE)
            & ~err_scr_nodes.removed
            & ~err_scr_nodes.added
        ][["issue", "origin", "flipped", "name"]]
        .drop_duplicates()
        .groupby(by=["issue", "origin", "flipped"], dropna=False)["name"]
        .apply("size")
        .to_frame("n_scr_err")
        .reset_index()
    )

    df_tree_features["n_scr_err"] = (
        df_tree_features.merge(_scr_err, on=["issue", "origin", "flipped"], how="left")[
            "n_scr_err"
        ]
        .fillna(0)
        .tolist()
    )

    _scr_add = (
        err_scr_nodes[(err_scr_nodes.type == SCRIPT_TYPE) & err_scr_nodes.added][
            ["issue", "origin", "flipped", "name"]
        ]
        .drop_duplicates()
        .groupby(by=["issue", "origin", "flipped"], dropna=False)["name"]
        .apply("size")
        .to_frame("n_scr_err_add")
        .reset_index()
    )

    df_tree_features["n_scr_err_add"] = (
        df_tree_features.merge(_scr_add, on=["issue", "origin", "flipped"], how="left")[
            "n_scr_err_add"
        ]
        .fillna(0)
        .tolist()
    )

    _scr_rem = (
        err_scr_nodes[(err_scr_nodes.type == SCRIPT_TYPE) & err_scr_nodes.removed][
            ["issue", "origin", "flipped", "name"]
        ]
        .drop_duplicates()
        .groupby(by=["issue", "origin", "flipped"], dropna=False)["name"]
        .apply("size")
        .to_frame("n_scr_err_rem")
        .reset_index()
    )

    df_tree_features["n_scr_err_rem"] = (
        df_tree_features.merge(_scr_rem, on=["issue", "origin", "flipped"], how="left")[
            "n_scr_err_rem"
        ]
        .fillna(0)
        .tolist()
    )
    
    # requests
    _req = (
        env_nodes[
            (env_nodes.type == REQUEST_TYPE)
            & ~env_nodes.removed
            & ~env_nodes.added
        ][["issue", "origin", "flipped", "name"]]
        .drop_duplicates()
        .groupby(by=["issue", "origin", "flipped"], dropna=False)["name"]
        .apply("size")
        .to_frame("n_req")
        .reset_index()
    )

    df_tree_features["n_req"] = (
        df_tree_features.merge(_req, on=["issue", "origin", "flipped"], how="left")[
            "n_req"
        ]
        .fillna(0)
        .tolist()
    )

    _req_add = (
        env_nodes[(env_nodes.type == REQUEST_TYPE) & env_nodes.added][
            ["issue", "origin", "flipped", "name"]
        ]
        .drop_duplicates()
        .groupby(by=["issue", "origin", "flipped"], dropna=False)["name"]
        .apply("size")
        .to_frame("n_req_add")
        .reset_index()
    )
    a = (
        df_tree_features.merge(_req_add, on=["issue", "origin", "flipped"], how="left")[
            "n_req_add"
        ]
        .fillna(0)
        .tolist()
    )

    df_tree_features["n_req_add"] = a

    _req_rem = (
        env_nodes[(env_nodes.type == REQUEST_TYPE) & env_nodes.removed][
            ["issue", "origin", "flipped", "name"]
        ]
        .drop_duplicates()
        .groupby(by=["issue", "origin", "flipped"], dropna=False)["name"]
        .apply("size")
        .to_frame("n_req_rem")
        .reset_index()
    )

    df_tree_features["n_req_rem"] = (
        df_tree_features.merge(_req_rem, on=["issue", "origin", "flipped"], how="left")[
            "n_req_rem"
        ]
        .fillna(0)
        .tolist()
    )

    LOGGER.debug("Global Features: done")

    return df_tree_features
