import pandas as pd
import plotly.graph_objects as go
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
from BreakageClassifier.code.features.utils import get_subtree


def _get_children(tree: pd.DataFrame, root: pd.Series) -> pd.DataFrame:
    
    
    if root.prev_id != -1:
        key, parent_key = "prev_id", "prev_parent"
    else:
        key, parent_key = "new_id", "new_parent"
    
    return tree[tree[parent_key] == root[key]].copy(deep=True)

def visualize_subtrees(subtrees: pd.DataFrame, root: pd.Series):
    subtree = get_subtree(subtrees, root).drop_duplicates()

    # i can get the children of the root
    
    # i want to plot the tree
    
    
    G = nx.DiGraph()
    print(root)
    for node in subtree.itertuples():
        print(node)
        if node.new_id != -1:
            G.add_node(node.new_id, label=node.tag)
            if node.new_id != root.new_id:
                G.add_edge(node.new_parent, node.new_id)
        else:
            G.add_node(node.prev_id, label=node.tag)
            if node.prev_id != root.prev_id:
                G.add_edge(node.prev_parent, node.prev_id)
            
    # show graph
    fig = go.Figure(
        data=[
            go.Scatter(
                x=[],
                y=[],
                mode="markers+text",
                textposition="bottom center",
                textfont_size=10,
                hoverinfo="none",
                marker=dict(color="LightSkyBlue", size=10),
            )
        ],
        layout=go.Layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
        ),
    )
    
    pos = graphviz_layout(G, prog="dot")
   
    for node in G.nodes():
        x, y = pos[str(node)]
        fig.add_annotation(
            x=x,
            y=y,
            xref="x",
            yref="y",
            text=G.nodes[node].get("label", ""),
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-30,
        )
    
    for edge in G.edges():
        x0, y0 = pos[str(edge[0])]
        x1, y1 = pos[str(edge[1])]
        fig.add_shape(
            type="line",
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            line=dict(color="LightSkyBlue", width=0.5),
        )
        
    fig.show()
    
    
    
    
     
    return G
