import re
import json
import traceback

import pandas as pd
from BreakageClassifier.code.graph.requests import get_callstack_origins_for_requests
from BreakageClassifier.code.utils import graph_node_id

from BreakageClassifier.code.graph.constants import (
    DOM_TYPE,
    ERROR_TYPE,
    INTERACT_ERROR_TYPE,
    INTERACT_RELATE_DOM_TYPE,
    INTERACTION_TYPE,
    JS_ELEM_SYMBOLS,
    JS_ERROR_SYMBOLS,
    JS_GET_ELEMENT_BY_ID,
    JS_GET_ELEMENTS_BY_CLASS_NAME,
    JS_GET_ELEMENTS_BY_TAG_NAME,
    JS_QUERY_SELECTOR,
    JS_QUERY_SELECTORS,
    REQUEST_TYPE,
    SCRIPT_RELATE_DOM_TYPE,
    SCRIPT_TYPE,
)
from BreakageClassifier.code.graph.utils import (
    extract_attributes_from_css,
    get_nodes_matching,
    replace_duplicates_with_count,
)

from ..logger import LOGGER


def convert_attr(row):
    """
    Function to create attributes for created elements.

    Args:
        row: Row of created elements DataFrame.
    Returns:
        attr: JSON string of attributes for created elements.
    """

    attr = {}
    try:
        attr["openwpm"] = json.loads(row["attributes"])["0"]["openwpm"]
        attr["subtype"] = row["subtype_list"]
        if row["script_loc_eval"] != "":
            attr["eval"] = True
        else:
            attr["eval"] = False
        attr = json.dumps(attr)
        return attr
    except Exception as e:
        LOGGER.warning("[ convert_attr ] : ERROR - ", exc_info=True)
        return json.dumps(attr)


def convert_subtype(arguments):
    """
    Function to obtain subtype of an element.

    Args:
        arguments: arguments column of javascript table for a created element.
    Returns:
        Sub-type fo created element (or emptry string in case of errors).
    """

    try:
        return json.loads(x)[0]
    except Exception as e:
        return ""


def get_tag(record, key):
    """
    Function to obtain the openwpm tag value.

    Args:
        record: Record to check tags.
        key: Key to get correct tag value.
    Returns:
        Tag value.
    """

    try:
        val = json.loads(record)

        if key == "fullopenwpm":
            openwpm = val.get("0").get("openwpm")
            return str(openwpm)
        else:
            return str(val.get(key))
    except Exception as e:
        return ""
    return ""


def find_parent_elem(src_elements, df_element):
    """
    Function to find parent element of .src JS elements.

    Args:
        src_element: DataFrame representation of src elements.
        df_element: DataFrame representation of created elements.
    Returns:
        result: Merged DataFrame representation linking created elements with src elements.
    """

    src_elements["new_attr"] = src_elements["attributes"].apply(
        get_tag, key="fullopenwpm"
    )
    df_element["new_attr"] = df_element["attr"].apply(get_tag, key="openwpm")
    result = src_elements.merge(
        df_element[["new_attr", "name"]], on="new_attr", how="left"
    )
    return result


def build_html_components(df_javascript):
    """
    Function to create HTML nodes and edges. This is limited since we
    don't capture all HTML behaviors -- we look at createElement and src JS calls.

    Args:
        df_javascript: DataFrame representation of OpenWPM's javascript table.
    Returns:
        df_js_nodes: DataFrame representation of HTML nodes
        df_js_edges: DataFrame representation of HTML edges
    """

    df_js_nodes = pd.DataFrame()
    df_js_edges = pd.DataFrame()

    try:
        # Find all created elements
        created_elements = df_javascript[
            df_javascript["symbol"] == "window.document.createElement"
        ].copy()

        df_element_nodes = pd.DataFrame(
            columns=["visit_id", "name", "top_level_url", "type", "attr"]
        )

        if len(created_elements) > 0:
            created_elements["name"] = created_elements.index.to_series().apply(
                lambda x: "Element_" + str(x)
            )
            created_elements["type"] = "Element"

            created_elements["subtype_list"] = created_elements["arguments"].apply(
                convert_subtype
            )
            created_elements["attr"] = created_elements.apply(convert_attr, axis=1)
            created_elements["action"] = "create"

            # Created Element nodes and edges (to be inserted)
            df_element_nodes = created_elements[
                ["visit_id", "name", "top_level_url", "type", "attr"]
            ]
            df_create_edges = created_elements[
                [
                    "visit_id",
                    "script_url",
                    "name",
                    "top_level_url",
                    "action",
                    "time_stamp",
                ]
            ]
            df_create_edges = df_create_edges.rename(
                columns={"script_url": "src", "name": "dst"}
            )

        src_elements = df_javascript[
            (df_javascript["symbol"].str.contains("Element.src"))
            & (df_javascript["operation"].str.contains("set"))
        ].copy()

        if len(src_elements) > 0:
            src_elements["type"] = "Request"
            src_elements = find_parent_elem(src_elements, df_element_nodes)
            src_elements["action"] = "setsrc"

            # Src Element nodes and edges (to be inserted)
            df_src_nodes = src_elements[
                ["visit_id", "value", "top_level_url", "type", "attributes"]
            ].copy()
            df_src_nodes = df_src_nodes.rename(
                columns={"value": "name", "attributes": "attr"}
            )
            df_src_nodes = df_src_nodes.dropna(subset=["name"])

            df_src_edges = src_elements[
                ["visit_id", "name", "value", "top_level_url", "action", "time_stamp"]
            ]
            df_src_edges = df_src_edges.dropna(subset=["name"])
            df_src_edges = df_src_edges.rename(columns={"name": "src", "value": "dst"})

            df_js_nodes = pd.concat([df_element_nodes, df_src_nodes]).drop_duplicates()
            df_js_nodes = df_js_nodes.drop(columns=["new_attr"])
            df_js_edges = pd.concat([df_create_edges, df_src_edges])

            df_js_edges["reqattr"] = "N/A"
            df_js_edges["respattr"] = "N/A"
            df_js_edges["response_status"] = "N/A"
            df_js_edges["attr"] = "N/A"

    except Exception as e:
        LOGGER.warning("Error in build_html_components:", exc_info=True)
        return df_js_nodes, df_js_edges

    return df_js_nodes, df_js_edges


def _script_to_node(script_url):
    return {"name": script_url, "type": SCRIPT_TYPE, "value": None}


def _request_to_node(request: pd.Series):
    return {"name": request.url, "type": REQUEST_TYPE, "value": None}


def _interaction_to_node(interaction):
    return {
        "name": str(interaction.timestamp),
        "type": INTERACTION_TYPE,
        "value": interaction.node_id,
    }


def _interaction_error_to_edge(
    js_event: pd.Series,
    interaction,
):
    return {
        "src": str(interaction.timestamp),
        "dst": js_event.script_url,
        "type": INTERACT_ERROR_TYPE,
        "value": "1",
        "n_events": js_event.n_events,
    }


def _request_to_edge(script: pd.Series, request: pd.Series):
    return {
        "src": script["name"],
        "dst": request.url,
        "type": REQUEST_TYPE,
        "value": None,
    }


def _elem_to_edges(js_event: pd.Series, source, type, df_dom):
    args = json.loads(js_event.arguments)

    tag, _id, classes, attributes = (None, None, [], dict())

    # TODO: attributes filters are not implemented

    only_one = False
    classes_one_of = False

    if js_event.symbol in JS_QUERY_SELECTORS:
        tag, _id, classes, attributes = extract_attributes_from_css(args[0])

        if js_event.symbol == JS_QUERY_SELECTOR:
            only_one = True

    elif js_event.symbol == JS_GET_ELEMENT_BY_ID:
        _id = args[0]
        only_one = True

    elif js_event.symbol == JS_GET_ELEMENTS_BY_TAG_NAME:
        tag = args[0]

    elif js_event.symbol == JS_GET_ELEMENTS_BY_CLASS_NAME:
        classes = args
        classes_one_of = True

    else:
        print("Symbol not expected! ", js_event)
        return pd.DataFrame(columns=["name", "type", "value"]), pd.DataFrame(
            columns=["src", "dst", "type", "value", "n_events"]
        )

    matches = set()

    if classes_one_of:
        for _class in classes:
            matches |= set(get_nodes_matching(df_dom, tag, _id, [_class]).values)

    else:
        matches = get_nodes_matching(df_dom, tag, _id, classes).values
        if only_one:
            matches = set([matches[0]]) if len(matches) else set()
        else:
            matches = set(matches)

    graph_nodes = []
    graph_edges = []

    browser_id = df_dom.iloc[0].browser_id
    visit_id = df_dom.iloc[0].visit_id

    for _, node_id in enumerate(matches):
        graph_nodes.append({"name": node_id, "type": DOM_TYPE, "value": None})
        graph_edges.append(
            {
                "src": source,
                "dst": graph_node_id(browser_id, visit_id, node_id),
                "type": type,
                "value": js_event["arguments"],  # what defines an error is
                "n_events": js_event.n_events,
            }
        )

    if not len(graph_nodes):
        graph_nodes = pd.DataFrame(columns=["name", "type", "value"])
    else:
        graph_nodes = pd.DataFrame(graph_nodes)

    if not len(graph_edges):
        graph_edges = pd.DataFrame(columns=["src", "dst", "type", "value", "n_events"])
    else:
        graph_edges = pd.DataFrame(graph_edges)

    return graph_nodes, graph_edges


def extract_js_edges(
    df_dom: pd.DataFrame,
    df_javascript: pd.DataFrame,
    df_interactions: pd.DataFrame,
    df_callstack: pd.DataFrame,
    df_responses: pd.DataFrame,
):
    # print(f"!original len(df_javascript)={len(df_javascript)}")
    _df_javascript = df_javascript[["script_url", "symbol", "arguments", "interaction"]]
    _df_javascript = replace_duplicates_with_count(_df_javascript)
    _df_javascript.rename(columns={"count": "n_events"}, inplace=True)
    # print(f"!original len(_df_javascript)={len(_df_javascript)}")

    if _df_javascript.empty:
        return pd.DataFrame(columns=["name", "type", "value"]), pd.DataFrame(
            columns=["src", "dst", "type", "value", "count"]
        )

    # get the scripts

    script_nodes = pd.DataFrame(
        [_script_to_node(n) for n in _df_javascript.script_url.unique()]
    )

    if not df_interactions.empty:
        interaction_nodes = df_interactions.apply(
            _interaction_to_node, result_type="expand", axis=1
        )

    else:
        interaction_nodes = pd.DataFrame(columns=["name", "type", "value"])

    elem_nodes = []
    elem_edges = []
    error_edges = []

    # finding the tree node from the css selector

    # print(f"!processing for interaction. len(interactions)={len(df_interactions)}")

    for _, interaction in df_interactions.iterrows():
        df_errors = _df_javascript[
            (_df_javascript.interaction == interaction.timestamp)
            & (_df_javascript.symbol.isin(JS_ERROR_SYMBOLS))
        ]

        if len(df_errors):
            _en = df_errors.apply(
                lambda row: _interaction_error_to_edge(row, interaction),
                result_type="expand",
                axis=1,
            )

            error_edges.append(_en)

            # now for the query nodes
        df_elems = _df_javascript[
            (_df_javascript.interaction == interaction.timestamp)
            & (_df_javascript.symbol.isin(JS_ELEM_SYMBOLS))
        ]

        if len(df_elems):
            _elem_nodes = []
            _elem_edges = []

            for _, event in df_elems.iterrows():
                _enodes, _eedges = _elem_to_edges(
                    event, str(interaction.timestamp), INTERACT_RELATE_DOM_TYPE, df_dom
                )
                _elem_nodes.append(_enodes)
                _elem_edges.append(_eedges)

            elem_nodes.append(pd.concat(_elem_nodes))
            elem_edges.append(pd.concat(_elem_edges))

    if len(error_edges):
        error_edges = pd.concat(error_edges)
    else:
        error_edges = pd.DataFrame(columns=["src", "dst", "type", "value", "n_events"])

    if len(elem_nodes):
        elem_nodes = pd.concat(elem_nodes)
    else:
        elem_nodes = pd.DataFrame(columns=["name", "type", "value"])

    if len(elem_edges):
        elem_edges = pd.concat(elem_edges)
    else:
        elem_edges = pd.DataFrame(columns=["src", "dst", "type", "value", "n_events"])

    # remaining errors not caused without interactions as loop edges
    _remaining_errors = _df_javascript[
        _df_javascript.symbol.isin(JS_ERROR_SYMBOLS)
        & _df_javascript.interaction.isna().values
    ]
    # _remaining_errors = df_errors[~df_errors["id"].isin(error_edges["value"].values)]

    if not _remaining_errors.empty:
        _remaining_errors = _remaining_errors.apply(
            lambda error: {
                "src": error.script_url,
                "dst": error.script_url,
                "type": ERROR_TYPE,
                "value": error.symbol,
                "n_events": error.n_events,
            },
            result_type="expand",
            axis=1,
        )

        error_edges = pd.concat([error_edges, _remaining_errors])

    # elem edges from scripts to elems

    df_elems = _df_javascript[_df_javascript.symbol.isin(JS_ELEM_SYMBOLS)]

    _elem_nodes = []
    _elem_edges = []

    for _, event in df_elems.iterrows():
        _enodes, _eedges = _elem_to_edges(
            event, event.script_url, SCRIPT_RELATE_DOM_TYPE, df_dom
        )
        _elem_nodes.append(_enodes)
        _elem_edges.append(_eedges)

    elem_edges = pd.concat([elem_edges, *_elem_edges])
    elem_nodes = pd.concat([elem_nodes, *_elem_nodes])

    edges = [
        error_edges,
        elem_edges,
    ]

    edges = pd.concat(edges)

    edges = replace_duplicates_with_count(edges, "n_events")

    nodes = [
        script_nodes,
        interaction_nodes,
        elem_nodes,
    ]

    nodes = pd.concat(nodes)

    # extract requests nodes and edges from javascript callstack
    df_request_origins = get_callstack_origins_for_requests(df_responses, df_callstack)

    _request_nodes = []
    _request_edges = []
    _request_node_urls = []

    for _, request in df_request_origins.iterrows():
        if request.url not in _request_node_urls:
            _request_nodes.append(_request_to_node(request))
            _request_node_urls.append(request.url)

    for _, script in script_nodes.iterrows():
        script_callstack_requests = df_request_origins[
            df_request_origins.origins.apply(lambda x: script["name"] in x)
        ]
        for _, request in script_callstack_requests.iterrows():
            _request_edges.append(_request_to_edge(script, request))

    request_nodes = (
        pd.DataFrame(_request_nodes)
        if len(_request_nodes)
        else pd.DataFrame(columns=["name", "type", "value"])
    )
    request_edges = (
        pd.DataFrame(_request_edges)
        if len(_request_edges)
        else pd.DataFrame(columns=["src", "dst", "type", "value", "count"])
    )

    nodes = pd.concat([nodes, request_nodes])
    edges = pd.concat([edges, request_edges])

    return (
        nodes.drop_duplicates().reset_index(drop=True)
        if len(nodes)
        else pd.DataFrame(columns=["name", "type", "value"]),
        edges.drop_duplicates().reset_index(drop=True)
        if len(edges)
        else pd.DataFrame(columns=["src", "dst", "type", "value", "count"]),
    )
