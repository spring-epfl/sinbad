import json
import numpy as np
import pandas as pd
from typing import Union
import tldextract
import re


def get_val_from_attributes(attributes: str, key: str):

    _attribute_list = json.loads(attributes)

    for d in _attribute_list:
        if d["key"] == key:
            return d["value"]

    return ""


def get_domain(url):

    assert isinstance(url, str), "not string"

    try:
        extract = tldextract.extract(url)
        return extract.domain + "." + extract.suffix

    except:
        return ""


def get_cookiedom_key(name, domain):

    """Function to get cookie key, a combination of the cookie name and domain.

    Args:
        name: cookie name
        domain: cookie domain
    Returns:
        cookie key
    """

    try:
        return name + "|$$|" + domain
    except:
        return name
    return name


def get_original_cookie_setters(df):

    """Function to get the first setter of a cookie.

    Args:
        df: DataFrame representation of all cookie sets.
    Returns:
        DataFrame representation of the cookie setter.
    """

    df_owners = {}
    df.sort_values("time_stamp", ascending=False, inplace=True)
    grouped = df.groupby(["visit_id", "dst"])
    rows_added = 0

    for name, group in grouped:
        if len(group) > 0:
            name_dict = {"visit_id": name[0], "dst": name[1]}
            final_dict = {**name_dict, **group.iloc[0].to_dict()}
            df_owners[rows_added] = final_dict
            rows_added += 1

    if rows_added > 0:
        df_owners = pd.DataFrame.from_dict(df_owners, "index")
        df_owners = df_owners[["visit_id", "dst", "src", "time_stamp"]]
        df_owners = df_owners.rename(
            columns={"dst": "name", "src": "setter", "time_stamp": "setting_time_stamp"}
        )
        return df_owners
    else:
        return pd.DataFrame(
            columns=["visit_id", "name", "setter", "setting_time_stamp"]
        )


def update_children_t(node, children):

    next_t = node["t_enter"]

    for i, _ in children.iterrows():
        children.at[i, "parent_id"] = node["id"]
        children.at[i, "t_enter"] = next_t + 1
        children.at[i, "t_leave"] = next_t + 2
        next_t += 2

    next_t += 1

    return children, next_t


def update_t_enter_leave_after_insert_subtree(tree, node, new_t_leave):

    # update all nodes that we leave after it in the tree visit
    for i, _ in tree[tree["t_leave"] >= node["t_leave"]].iterrows():
        tree.at[i, "t_leave"] += new_t_leave - node["t_leave"] + 1

    # update all nodes that were after it in the tree visit
    for i, _ in tree[tree["t_enter"] > node["t_leave"]].iterrows():
        tree.at[i, "t_enter"] += new_t_leave - node["t_leave"] + 1

    return tree


def extract_attributes_from_css(css_selector):
    pattern = r"(?P<tag>\*|\w+)?(?:#(?P<id>\w+))?(?:\.(?P<classes>\w+(?:\.\w+)*))?(?P<attributes>\[.+?\])?"
    match = re.match(pattern, css_selector)

    tag_type = match.group("tag") if match.group("tag") else "any"
    element_id = match.group("id") if match.group("id") else None
    classes = match.group("classes").split(".") if match.group("classes") else []

    attributes = {}
    if match.group("attributes"):
        attr_pattern = r'\[(\w+)(?:=(["\'])(.*?)\2)?\]'
        attr_matches = re.findall(attr_pattern, match.group("attributes"))
        for attr_name, _, attr_value in attr_matches:
            attributes[attr_name] = attr_value

    return tag_type, element_id, classes, attributes


def get_nodes_matching(df_dom, tag, id, classes) -> pd.Series:

    filters = True
    
    # TODO: toggle on/off
    if tag == "*":
        filters = False

    if tag and tag != "*":
        filters &= df_dom.nodeName == tag

    if id:
        filters &= df_dom.attributes.apply(
            lambda attributes: get_val_from_attributes(attributes, "id") == id
        )

    if classes and len(classes):

        filters &= df_dom.attributes.apply(
            lambda attributes: set(
                get_val_from_attributes(attributes, "class").split(" ")
            )
            == classes
        )

    if not isinstance(filters, bool):
        return df_dom[filters]["id"]

    if filters == True:
        return df_dom["id"]
    else:
        return pd.DataFrame(columns=df_dom.columns)


def replace_duplicates_with_count(df: pd.DataFrame, weight_col=None):
    
    if len(df) == 0:
        return pd.DataFrame(columns = list(df.columns) + ["count"])
    
    if not weight_col:
        duplicate_counts = (
            df.groupby(df.columns.tolist()).apply("size").to_frame("count").reset_index()
        )
    
    else:
        
        cols = list(set(df.columns.tolist()) - {weight_col, })
        
        duplicate_counts = (
            df.groupby(cols).apply(lambda _df: _df[weight_col].sum()).to_frame("count").reset_index()
        )

    return duplicate_counts