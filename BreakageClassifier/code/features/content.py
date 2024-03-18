from .utils import get_subtree, time_it
from six.moves.urllib.parse import urlparse, parse_qs
from sklearn import preprocessing
import re
import json
import pandas as pd
from tqdm import tqdm
from ..graph import get_val_from_attributes

from ..logger import LOGGER

tqdm.pandas()


def _get_url_features_subtree(root: pd.Series, df_edits: pd.DataFrame, cols: list):
    """
    Function to extract URL features.
    """
    # url, node_dict

    subtree = get_subtree(df_edits, root)
    subtree_with_request_rows = subtree[(subtree["party"] != "N/A")]

    def _label_one_row(row: pd.Series):
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
        keyword_char = [
            ".",
            "/",
            "&",
            "=",
            ";",
            "-",
            "_",
            "/",
            "*",
            "^",
            "?",
            ";",
            "|",
            ",",
        ]
        screen_resolution = [
            "screenheight",
            "screenwidth",
            "browserheight",
            "browserwidth",
            "screendensity",
            "screen_res",
            "screen_param",
            "screenresolution",
            "browsertimeoffset",
        ]

        attrs = row.prev_attr or row.new_attr

        try:
            url = get_val_from_attributes(attrs, "url")
            parsed_url = urlparse()
            query = parsed_url.query
            params = parsed_url.params
            is_valid_qs = 1
            base_domain = row.domain

        except:
            url = ""
            base_domain = ""
            query = ""
            params = ""
            is_valid_qs = 0

        parsed_query = parse_qs(query)
        parsed_params = parse_qs(params)
        num_url_queries = len(parsed_query)
        num_url_params = len(parsed_params)
        num_id_in_query_field = len([x for x in parsed_query.keys() if "id" in x])
        num_id_in_param_field = len([x for x in parsed_params.keys() if "id" in x])

        is_third_party = 1 if row.party == "third" else 0

        semicolon_in_query = 0
        semicolon_in_params = 0
        base_domain_in_query = 0
        if len(base_domain) > 0 and base_domain in query:
            base_domain_in_query = 1
        if ";" in query:
            semicolon_in_query = 1
        if ";" in params:
            semicolon_in_params = 1

        screen_size_present = 0
        for screen_key in screen_resolution:
            if screen_key in query.lower() or screen_key in params.lower():
                screen_size_present = 1
                break
        ad_size_present = 0
        ad_size_in_qs_present = 0
        pattern = "\d{2,4}[xX]\d{2,4}"
        if re.compile(pattern).search(url):
            ad_size_present = 1
        if re.compile(pattern).search(query):
            ad_size_in_qs_present = 1

        keyword_char_present = 0
        keyword_raw_present = 0

        for key in keyword_raw:
            key_matches = [m.start() for m in re.finditer(key, url, re.I)]

            for key_match in key_matches:
                keyword_raw_present = 1
                if url[key_match - 1] in keyword_char:
                    keyword_char_present = 1
                    break
            if keyword_char_present == 1:
                break

        return (
            is_valid_qs,
            num_url_queries,
            num_url_params,
            num_id_in_query_field,
            num_id_in_param_field,
            is_third_party,
            base_domain_in_query,
            semicolon_in_query,
            semicolon_in_params,
            screen_size_present,
            ad_size_present,
            ad_size_in_qs_present,
            keyword_raw_present,
            keyword_char_present,
        )

    # url_feature_names = [
    #     "is_valid_qs",
    #     "num_url_queries",
    #     "num_url_params",
    #     "num_id_in_query_field",
    #     "num_id_in_param_field",
    #     "is_third_party",
    #     "base_domain_in_query",
    #     "semicolon_in_query",
    #     "semicolon_in_params",
    #     "screen_size_present",
    #     "ad_size_present",
    #     "ad_size_in_qs_present",
    #     "keyword_raw_present",
    #     "keyword_char_present",
    # ]

    subtree_with_request_rows_features = subtree_with_request_rows.copy()

    if len(subtree_with_request_rows):
        subtree_with_request_rows_features[cols] = subtree_with_request_rows.apply(
            _label_one_row, axis=1, result_type="expand"
        )

        N = len(subtree_with_request_rows_features)

        return (
            subtree_with_request_rows_features.is_valid_qs.sum() / N,
            subtree_with_request_rows_features.num_url_queries.sum() / N,
            subtree_with_request_rows_features.num_url_params.sum() / N,
            subtree_with_request_rows_features.num_id_in_query_field.sum() / N,
            subtree_with_request_rows_features.num_id_in_param_field.sum() / N,
            subtree_with_request_rows_features.is_third_party.sum() / N,
            subtree_with_request_rows_features.base_domain_in_query.sum() / N,
            subtree_with_request_rows_features.semicolon_in_query.sum() / N,
            subtree_with_request_rows_features.semicolon_in_params.sum() / N,
            subtree_with_request_rows_features.screen_size_present.sum() / N,
            subtree_with_request_rows_features.ad_size_present.sum() / N,
            subtree_with_request_rows_features.ad_size_in_qs_present.sum() / N,
            subtree_with_request_rows_features.keyword_raw_present.sum() / N,
            subtree_with_request_rows_features.keyword_char_present.sum() / N,
            N,
        )

    else:
        return [0] * (len(cols) + 1)


def get_node_features(node_name, node_dict, le):
    """
    Function to extract node features.

    Args:
      node_name: URL of node
      node_dict: Attribute of node (domain/content policy type/top level URL)
      le: LabelEncoding for node type
    Returns:
      node_features: node feature values
      node_feature_names: node feature names
    """

    node_features = []
    content_policy_type = None
    is_subdomain = 0
    url_length = 0
    node_type = ""

    try:
        url_length = len(node_name)
        node_type = le.transform([node_dict["type"]])[0]
        attr = node_dict["attr"]
        top_level_domain = node_dict["top_level_domain"]
        domain = node_dict["domain"]

        if "content_policy_type" in attr:
            content_policy_type = json.loads(attr)["content_policy_type"]
        if top_level_domain and domain:
            if domain == top_level_domain:
                is_subdomain = 1
        node_features = [node_type, content_policy_type, url_length, is_subdomain]

    except Exception as e:
        LOGGER.warning("Error node features:", exc_info=True)
        node_features = [node_type, content_policy_type, url_length, is_subdomain]

    node_feature_names = [
        "node_type",
        "content_policy_type",
        "url_length",
        "is_subdomain",
    ]

    return node_features, node_feature_names


def get_content_features(G, df_graph, node):
    """
    Function to extract content features. This function calls
    the other functions to extract different types of content features.

    Args:
      G: networkX graph
      df_graph: DataFrame representation of graph
      node: URL of node
    Returns:
      all_features: content feature values
      all_feature_names: content feature names
    """

    le = preprocessing.LabelEncoder()
    le.fit(["Request", "Script", "Document", "Element", "Storage"])

    all_features = []
    all_feature_names = []
    node_features, node_feature_names = get_node_features(node, G.nodes[node], le)
    url_features, url_feature_names = get_url_features(node, G.nodes[node])
    all_features = node_features + url_features
    all_feature_names = node_feature_names + url_feature_names

    return all_features, all_feature_names


def _get_counts_for_subtree(root: pd.Series, df_edits: pd.DataFrame, cols: list):
    layout_tags = [
        "span",
        "div",
        "li",
        "header",
        "section",
        "footer",
        "ul",
        "iframe",
        "article",
        "main",
        "aside",
    ]

    text_tags = [
        "#text",
        "h4",
        "h3",
        "h2",
        "h1",
        "p",
        "i",
        "br",
        "strong",
    ]

    io_tags = [
        "a",
        "nav",
        "img",
        "button",
        "form",
        "input",
        "picture",
        "source",
        "svg",
    ]

    def _get_label(row):
        if row.tag in layout_tags:
            pre = "lt"
        elif row.tag in text_tags:
            pre = "txt"
        elif row.tag in io_tags:
            pre = "io"
        else:
            pre = "oth"

        if row.removed:
            suf = "rem"
        elif row.added:
            suf = "add"
        elif row.edited:
            suf = "ed"
        else:
            suf = "no"

        return pre + "_" + suf

    def _is_visible(row):
        try:
            prev_visible = json.loads(row.prev_vc)["is_visible"]
        except:
            prev_visible = -1

        try:
            new_visible = json.loads(row.new_vc)["is_visible"]
        except:
            new_visible = -1

        if prev_visible == -1 and new_visible == -1:
            return float("nan"), float("nan"), float("nan")
        elif prev_visible == -1:
            return False, new_visible, False
        elif new_visible == -1:
            return prev_visible, False, prev_visible
        else:
            # true if became visible
            return (
                prev_visible and new_visible,
                new_visible and not prev_visible,
                prev_visible and not new_visible,
            )

    subtree = get_subtree(df_edits, root)
    _counts: pd.Series = subtree.apply(_get_label, axis=1).value_counts()

    _counts = _counts.to_dict()

    _visible = subtree.apply(_is_visible, axis=1).tolist()

    _visible = pd.DataFrame(_visible, columns=["both", "new", "prev"]).dropna()

    _counts["n_visible"] = int(_visible["both"].sum())
    _counts["n_visible_add"] = int(_visible["new"].sum())
    _counts["n_visible_rem"] = int(_visible["prev"].sum())

    total = sum(_counts.values())
    total = max(total, 1)

    _counts["n_salient"] = subtree["saliency"].sum()
    _counts["n_salient_rem"] = subtree[subtree["removed"]]["saliency"].sum()
    _counts["n_salient_add"] = subtree[subtree["added"]]["saliency"].sum()
    _counts["n_salient_ed"] = subtree[subtree["edited"]]["saliency"].sum()

    elems = ["script", "iframe"]

    for elem in elems:
        _counts[f"{elem}_no"] = subtree[subtree["tag"] == elem].shape[0]
        _counts[f"{elem}_add"] = subtree[
            (subtree["tag"] == elem) & (subtree["added"])
        ].shape[0]
        _counts[f"{elem}_rem"] = subtree[
            (subtree["tag"] == elem) & (subtree["removed"])
        ].shape[0]

    counts = {k: _counts.get(k, 0) for k in cols}
    counts["n_nodes"] = total

    return pd.Series(counts)


def get_element_counts(features: pd.DataFrame, df_edits: pd.DataFrame):
    """compute the counts of the element categories features. it adds them inplace to the features dataframe.

    Args:
        features (pd.DataFrame): dataframe of roots with features and will be edited by adding the new features.
        df_edits (pd.DataFrame): the edits trees dataframe.
    """

    cols = [
        "lt_add",
        "lt_rem",
        "lt_ed",
        "lt_no",
        "txt_add",
        "txt_rem",
        "txt_ed",
        "txt_no",
        "io_add",
        "io_rem",
        "io_ed",
        "io_no",
        "oth_add",
        "oth_rem",
        "oth_ed",
        "oth_no",
        "scr_no",
        "scr_add",
        "scr_rem",
        "iframe_no",
        "iframe_add",
        "iframe_rem",
        "n_visible",
        "n_visible_add",
        "n_visible_rem",
        "n_salient_add",
        "n_salient_rem",
        "n_salient_ed",
        "n_salient",
        "n_nodes",
    ]

    features[cols] = features.progress_apply(
        lambda row: _get_counts_for_subtree(row, df_edits, cols), axis=1
    )


def get_url_features(features: pd.DataFrame, df_edits: pd.DataFrame):
    cols = [
        "is_valid_qs",
        "num_url_queries",
        "num_url_params",
        "num_id_in_query_field",
        "num_id_in_param_field",
        "is_third_party",
        "base_domain_in_query",
        "semicolon_in_query",
        "semicolon_in_params",
        "screen_size_present",
        "ad_size_present",
        "ad_size_in_qs_present",
        "keyword_raw_present",
        "keyword_char_present",
        "num_requests",
    ]

    features[cols] = features.progress_apply(
        lambda row: _get_url_features_subtree(row, df_edits, cols[:-1]),
        axis=1,
        result_type="expand",
    )


def _get_visual_features_subtree(
    root: pd.Series, df_edits: pd.DataFrame, salient_nodes, cols: list
):
    vals = {}

    size_prev = 0
    size_new = 0
    x_prev = 0
    x_new = 0
    y_prev = 0
    y_new = 0

    if isinstance(root.prev_vc, str) and "bounds" in root.prev_vc:
        prev_vc = json.loads(root.prev_vc)

        size_prev = prev_vc["bounds"]["width"] * prev_vc["bounds"]["height"]
        x_prev = prev_vc["bounds"]["x"]
        y_prev = prev_vc["bounds"]["y"]

    if isinstance(root.new_vc, str) and "bounds" in root.new_vc:
        new_vc = json.loads(root.new_vc)
        size_new = new_vc["bounds"]["width"] * new_vc["bounds"]["height"]
        x_new = new_vc["bounds"]["x"]
        y_new = new_vc["bounds"]["y"]

    vals["max_size"] = max(size_prev, size_new)
    vals["min_size"] = min(size_prev, size_new)

    vals["size"] = size_prev
    vals["d_size"] = size_new - size_prev

    vals["pos_x"] = x_prev
    vals["pos_y"] = y_prev
    vals["d_pos_x"] = x_new - x_prev
    vals["d_pos_y"] = y_new - y_prev
    vals["d_pos"] = (vals["d_pos_x"] ** 2 + vals["d_pos_y"] ** 2) ** 0.5

    contained_prev = []
    contained_after = []

    for _, salient_node in salient_nodes[salient_nodes.issue == root.issue].iterrows():
        salient_vc = json.loads(salient_node.visual_cues)
        x = salient_vc["bounds"]["x"]
        y = salient_vc["bounds"]["y"]
        width = salient_vc["bounds"]["width"]
        height = salient_vc["bounds"]["height"]

        if (
            x_prev <= x
            and x + width <= x_prev + size_prev
            and y_prev <= y
            and y + height <= y_prev + size_prev
        ):
            contained_prev.append(salient_node.id)

        if (
            x_new <= x
            and x + width <= x_new + size_new
            and y_new <= y
            and y + height <= y_new + size_new
        ):
            contained_after.append(salient_node.id)

    vals["n_salient_covered"] = len(contained_prev)
    vals["dn_salient_covered"] = len(contained_after) - len(contained_prev)

    return pd.Series({k: vals.get(k, 0) for k in cols})


def get_visual_features(features: pd.DataFrame, df_edits: pd.DataFrame, salient_nodes):
    cols = [
        "max_size",
        "min_size",
        "size",
        "d_size",
        "pos_x",
        "pos_y",
        "d_pos_x",
        "d_pos_y",
        "d_pos",
        "n_salient_covered",
        "dn_salient_covered",
    ]

    features[cols] = features.progress_apply(
        lambda row: _get_visual_features_subtree(row, df_edits, salient_nodes, cols),
        axis=1,
        result_type="expand",
    )


@time_it
def get_dom_edits_content_features(df_edits: pd.DataFrame, salient_nodes):
    """get the dom edits content features.

    Args:
        df_edits (pd.DataFrame): the tree of edits dataframe.

    Returns:
        pd.DataFrame: the roots dataframe containing content features.
    """

    features_df = df_edits[df_edits["is_root"] == True].copy(deep=True)

    # get the various content features
    LOGGER.debug("Content Features: Element Counts")

    get_element_counts(features_df, df_edits)

    LOGGER.debug("Content Features: URL Features")

    get_url_features(features_df, df_edits)

    LOGGER.debug("Visual Features: Subtree Visual Features")

    get_visual_features(features_df, df_edits, salient_nodes)

    return features_df
