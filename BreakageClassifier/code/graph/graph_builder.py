import json
import networkx as nx
import numpy as np
import pandas as pd


def build_networkx_graph(pdf):
    """
    Function to build a networkX graph from a Pandas DataFrame.

    Args:
        pdf: DataFrame of nodes and edges.
    Returns:
        G: networkX graph.

    This functions does the following:

    1. Selects nodes and edges.
    2. Processes node attributes.
    3. Creates graph from edges.
    4. Updates node attributes in graph.
    """

    df_nodes = pdf[(pdf["graph_attr"] == "Node") | (pdf["graph_attr"] == "NodeWG")]
    df_edges = pdf[(pdf["graph_attr"] == "Edge") | (pdf["graph_attr"] == "EdgeWG")]
    df_nodes = df_nodes.groupby(["visit_id", "name"], as_index=False).agg(
        {
            "type": lambda x: list(x),
            "attr": lambda x: list(x),
            "domain": lambda x: list(x)[0],
            "top_level_domain": lambda x: list(x)[0],
        }
    )

    def modify_type(orig_type) -> str:
        orig_type = list(set(orig_type))
        if len(orig_type) == 1:
            return orig_type[0]

        new_type = "Request"
        if "Script" in orig_type:
            new_type = "Script"
        elif "Document" in orig_type:
            new_type = "Document"
        elif "Element" in orig_type:
            new_type = "Element"
        return new_type

    def modify_attr(orig_attr):
        orig_attr = np.array(list(set(orig_attr)))
        if len(orig_attr) == 1:
            return orig_attr[0]

        for item in orig_attr:
            if item and "top_level_url" in item:
                return json.loads(item)
        return ""

    df_nodes["type"] = df_nodes["type"].apply(modify_type)
    df_nodes["attr"] = df_nodes["attr"].apply(modify_attr)
    networkx_graph = nx.from_pandas_edgelist(
        df_edges, source="src", target="dst", edge_attr=True, create_using=nx.DiGraph()
    )
    node_dict = df_nodes.set_index("name").to_dict("index")
    nx.set_node_attributes(networkx_graph, node_dict)

    return networkx_graph


def dom_edits_tree_to_networkx_graph(root, tree_df):
    def __get_node_id(row):
        return (
            curr.prev_vid,
            curr.prev_id,
            curr.new_vid,
            curr.new_id,
        )

    edges = []
    nodes = set()

    Q = [root]

    while len(Q):

        curr = Q.pop(0)

        curr_id = __get_node_id(curr)

        if curr_id in nodes:
            continue

        nodes.add(curr_id)

        children = tree_df[
            (
                (~tree_df["prev_vid"].isnull())
                & (tree_df["prev_vid"] == curr.prev_vid)
                & (tree_df["prev_parent"] == curr.prev_id)
                & (tree_df["is_breaking"] == curr.is_breaking)
            )
            | (
                (~tree_df["new_vid"].isnull())
                & (tree_df["new_vid"] == curr.new_vid)
                & (tree_df["new_parent"] == curr.new_id)
                & (tree_df["is_breaking"] == curr.is_breaking)
            )
        ].drop_duplicates()

        for _, child in children.iterrows():

            edge = (
                curr_id,
                __get_node_id(child),
            )

            edges.append(edge)

            Q.append(child)

    grph: nx.Graph = nx.from_edgelist(edges)
    grph.add_node(__get_node_id(root))
    id_cols = ["prev_vid", "prev_id", "new_vid", "new_id"]
    node_dict = (
        tree_df.drop_duplicates(subset=id_cols).set_index(id_cols).to_dict("index")
    )

    nx.set_node_attributes(grph, node_dict)

    return grph


# tests
if __name__ == "__main__":

    df = pd.read_csv("../out-test1/edits.csv")

    df["is_root"].unique()

    for i, root in df[df["is_root"]].iterrows():
        dom_edits_tree_to_networkx_graph(root, df)
