import json
from pathlib import Path
import uuid
from .utils import (
    update_children_t,
    update_t_enter_leave_after_insert_subtree,
)
from .database import Database
import pandas as pd
from .constants import REQUEST_TYPE


def _resp_to_node(resp: pd.Series):

    return {
        'browser_id': resp.browser_id,
        'visit_id': resp.visit_id,
        'id': str(uuid.uuid4()),
        "nodeName":resp.method.lower() + "-request",
        "type":REQUEST_TYPE,
        "attributes": json.dumps(
            [
                {"key": "id", "value": f"{resp.method}:{resp.url}"},
                {"key": "url", "value": resp.url},
            ]
        ),
        "visual_cues": None,
        "parent_id": resp.id,
        "t_enter": None,
        "t_leave": None,
        "saliency": None,
        "block": None,
    }


def _get_src_from_attrs(attributes):

    attrs = json.loads(attributes)
    attrs = dict([(x["key"], x["value"]) for x in attrs])

    # in case it is the <object> tag they use data attribute as source
    return attrs.get("src", attrs.get("data", None))


def _augment_dom_with_src_requests(df_dom: pd.DataFrame, df_responses: pd.DataFrame):
    
    attr_keys = ["\"src\"", "\"data\""]

    elem_src: pd.DataFrame = df_dom[
        (df_dom["attributes"].str.contains("|".join(attr_keys)))
    ][["browser_id", "visit_id", "id", "nodeName", "attributes"]].copy()

    elem_src["src"] = elem_src["attributes"].apply(_get_src_from_attrs)
    elem_src.drop(columns="attributes", inplace=True)
    elem_src.dropna(inplace=True)
    
    resp_to_el = elem_src.merge(df_responses, left_on="src", right_on="url", how="left")
    resp_to_el.dropna(inplace=True)
    

    # we only care if we got a response, means the adblocker let it pass
    resp_to_el.drop(columns=["src", "frame_id", "timestamp", "time_stamp"], inplace=True)
    resp_to_el.drop_duplicates(keep="first", inplace=True)

    resp_to_el_nodes = pd.DataFrame(columns=df_dom.columns)

    if not resp_to_el.empty:
        resp_to_el_nodes = resp_to_el.apply(
            _resp_to_node, result_type="expand", axis=1
        )

    for parent_id in set(resp_to_el_nodes["parent_id"]):

        children = resp_to_el_nodes[resp_to_el_nodes["parent_id"] == parent_id]
        parent = df_dom[df_dom["id"] == parent_id].iloc[0]

        children, new_t_leave = update_children_t(parent, children)

        # make sure the parent and children are updated
        resp_to_el_nodes.loc[resp_to_el_nodes["parent_id"] == parent_id] = children

        df_dom = update_t_enter_leave_after_insert_subtree(
            df_dom, parent, new_t_leave - 1
        )

        df_dom = pd.concat([df_dom, children])

    return df_dom


def _augment_dom_with_iframe_requests(df_dom: pd.DataFrame, df_responses: pd.DataFrame):

    elem_iframe: pd.DataFrame = df_dom[
        df_dom["nodeName"].isin(["iframe", "embed", "object"])
    ][["browser_id", "visit_id", "id", "attributes", "t_enter", "t_leave"]].copy()

    elem_iframe["src"] = elem_iframe["attributes"].apply(_get_src_from_attrs)
    elem_iframe.drop(columns="attributes", inplace=True)
    elem_iframe.dropna(inplace=True)

    elem_iframe = elem_iframe.merge(
        df_responses, left_on="src", right_on="url", how="left"
    )
    elem_iframe.dropna(inplace=True)

    for _, iframe in elem_iframe.iterrows():

        # for each frame get all requests
        iframe_requests = df_responses[
            df_responses["frame_id"] == iframe.frame_id
        ].copy()
        iframe_requests.loc[:, "id"] = iframe.id
        iframe_requests.loc[:, "browser_id"] = iframe.browser_id
        iframe_requests.loc[:, "visit_id"] = iframe.visit_id

        # make sure we just capture the uncaptured requests to iframes not to their children
        iframe_requests = iframe_requests[
            ~(
                iframe_requests.apply(
                    lambda x: any(
                        df_dom["attributes"].str.contains(
                            x.method + ":" + x.url, regex=False
                        )
                    ),
                    axis=1,
                )
            )
        ]

        iframe_requests.drop_duplicates(keep="first", inplace=True)

        iframe_requests_nodes = pd.DataFrame(columns=df_dom.columns).apply(
            _resp_to_node, result_type="expand", axis=1
        )

        iframe_requests_nodes, new_t_leave = update_children_t(
            iframe, iframe_requests_nodes
        )

        df_dom = update_t_enter_leave_after_insert_subtree(
            df_dom, iframe, new_t_leave - 1
        )

        df_dom = pd.concat([df_dom, iframe_requests_nodes])

    return df_dom

def process_callstack(s):
    
    stack = []
    
    for line in s.splitlines():
        _, *url = line.split("@")
        
        url = "@".join(url)
        
        *url, _, _ = url.split(":")
        url = ":".join(url)
        
        stack.append(
            url
            )
    
    return list(set(stack))


def get_callstack_origins_for_requests(df_responses: pd.DataFrame, df_callstacks: pd.DataFrame):
    
    
    df_request_origins = df_callstacks[['request_id', 'call_stack']].merge(df_responses, on='request_id')
    df_request_origins['origins'] = df_request_origins['call_stack'].apply(process_callstack)
    df_request_origins.drop(columns=['call_stack'], inplace=True)
    
    return df_request_origins


def augment_dom_with_requests(df_dom, df_responses):

    # elements with src

    df_dom = _augment_dom_with_src_requests(df_dom, df_responses)

    # iframes

    df_dom = _augment_dom_with_iframe_requests(df_dom, df_responses)
    
    return df_dom


if __name__ == "__main__":

    # TESTING augment_dom_with_requests

    with Database(
        Path("../crawl/datadir-error-test/crawl-data.sqlite"),
        Path("../crawl/datadir-error-test/experiments.csv"),
    ) as database:

        visits = database.sites_visits()
        visit = visits.iloc[2]

        df_responses = database.get_http_responses(visit.visit_id_a)
        df_callstacks = database.get_callstacks(visit.visit_id_a)
       
        df_dom = database.get_dom_from_visit_id(visit.visit_id_a)

        out = augment_dom_with_requests(df_dom, df_responses, df_callstacks)
        print(out[out["type"] == REQUEST_TYPE])
