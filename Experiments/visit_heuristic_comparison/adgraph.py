from pathlib import Path
import re
import pandas as pd
import requests
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from storage_dump.storage import DataframeCSVStorageController

from tqdm import tqdm
from BreakageClassifier.code.classification.webpage_models import (
    ModelPipeline,
    Preprocessor,
)

from BreakageClassifier.code.graph.database import Database
from BreakageClassifier.code.run import GraphExtractionError
from imblearn.over_sampling import SMOTE

JS_NAVIGATOR = [
    "window.navigator.userAgent",
    "window.navigator.appVersion",
    "window.navigator.vendor",
    "window.navigator.platform",
    "window.navigator.doNotTrack",
    "window.navigator.language",
    "window.navigator.languages",
    "window.navigator.maxTouchPoints",
    "window.navigator.hardwareConcurrency",
    "window.navigator.deviceMemory",
    "window.navigator.connection",
    "window.navigator.plugins",
    "window.navigator.mimeTypes",
    "window.navigator.cookieEnabled",
    "window.navigator.onLine",
    "window.navigator.geolocation",
    "window.navigator.serviceWorker",
    "window.navigator.permissions",
    "window.navigator.mediaDevices",
    "window.navigator.credentials",
    "window.navigator.keyboard",
    "window.navigator.usb",
    "window.navigator.bluetooth",
    "window.navigator.presentation",
    "window.navigator.xr",
    "window.navigator.storage",
    "window.navigator.webkitTemporaryStorage",
    "window.navigator.webkitPersistentStorage",
    "window.navigator.storageQuota",
    "window.navigator.webkitPointer",
    "window.navigator.webkitGamepads",
    "window.navigator.webkitGetGamepads",
    "window.navigator.webkitGetUserMedia",
    "window.navigator.webkitPersistentStorage",
    "window.navigator.webkitTemporaryStorage",
    "window.navigator.webkitPointer",
    "window.navigator.webkitGamepads",
    "window.navigator.webkitGetGamepads",
    "window.navigator.webkitGetUserMedia",
]
JS_ALTER_DOM = [
    "window.document.createElement",
    "window.document.createElementNS",
    "window.document.createDocumentFragment",
    "window.document.createTextNode",
    "window.document.createComment",
    "window.document.createProcessingInstruction",
    "window.document.importNode",
    "window.document.adoptNode",
    "window.document.createAttribute",
    "window.document.createAttributeNS",
    "window.document.createEvent",
    "window.document.createRange",
    "window.document.createNodeIterator",
]
JS_INSERT_DOM = [
    "window.document.appendChild",
    "window.document.insertBefore",
    "window.document.replaceChild",
    "window.document.removeChild",
    "window.document.write",
    "window.document.writeln",
]

JS_DELETE_DOM = [
    "window.document.removeChild",
    "window.document.write",
    "window.document.writeln",
]

JS_FETCH_OR_EVAL = [
    "window.fetch",
    "window.eval",
]

JS_SESSION_STORAGE = [
    "window.sessionStorage",
    "window.sessionStorage.getItem",
    "window.sessionStorage.setItem",
    "window.sessionStorage.removeItem",
    "window.sessionStorage.clear",
]

JS_COOKIE = [
    "window.document.cookie",
    "window.document.cookie=",
]

JS_LOCAL_STORAGE = [
    "window.localStorage",
    "window.localStorage.getItem",
    "window.localStorage.setItem",
    "window.localStorage.removeItem",
    "window.localStorage.clear",
]

JS_WEBGL = [
    "window.WebGLRenderingContext",
    "window.WebGLRenderingContext.getExtension",
    "window.WebGLRenderingContext.getSupportedExtensions",
    "window.WebGLRenderingContext.getParameter",
    "window.WebGLRenderingContext.getShaderPrecisionFormat",
    "window.WebGLRenderingContext.getContextAttributes",
]

JS_API = [
    "window.postMessage",
    "window.open",
    "window.close",
    "window.print",
    "window.requestAnimationFrame",
    "window.cancelAnimationFrame",
    "window.requestIdleCallback",
    "window.cancelIdleCallback",
    "window.setTimeout",
    "window.clearTimeout",
    "window.setInterval",
    "window.clearInterval",
    "window.setImmediate",
    "window.clearImmediate",
    "window.requestFileSystem",
    "window.webkitRequestFileSystem",
]

JS_EVENT = [
    "window.addEventListener",
    "window.removeEventListener",
    "window.dispatchEvent",
]

JS_SCREEN = [
    "window.screen",
]


def _get_content_length_from_headers(headers):
    reg = r"(?<=content-length\",\")[0-9]*"

    groups = re.search(reg, headers)

    if groups is None:
        return 0

    return int(groups.group(0))


def req_features(
    df_request_response_in,
    callstacks_in,
    df_request_response_out,
    callstacks_out,
):
    features = {
        # 3 % of sub-document requests blocked
        "r_reqs_subdoc_rem": 0,
        # 1 Δ in bytes sent over network after blocking
        "req_bytes_sent_rem": 0,
        # 2 size of resources directly blocked
        "res_size_rem": 0,
        # 24 # of resources blocked (direct or indirect)
        "n_res_rem": 0,
        # 31 % of network resources that were blocked
        "r_res_rem": 0,
    }

    # 3
    iframe_req_in = set(
        df_request_response_in[
            df_request_response_in.resource_type.isin(["sub_frame", "object"])
        ].url.unique()
    )

    iframe_req_out = set(
        df_request_response_out[
            df_request_response_out.resource_type.isin(["sub_frame", "object"])
        ].url.unique()
    )

    features["r_reqs_subdoc_rem"] = (
        max(len(iframe_req_in - iframe_req_out) / len(iframe_req_in), 0)
        if len(iframe_req_in) > 0
        else 0
    )

    # 1
    content_length_in = df_request_response_in.resp_headers.apply(
        lambda x: _get_content_length_from_headers(x)
    )

    content_length_out = df_request_response_out.resp_headers.apply(
        lambda x: _get_content_length_from_headers(x)
    )

    features["req_bytes_sent_rem"] = max(
        content_length_in.sum() - content_length_out.sum(), 0
    )

    # 2
    # basically requests with no callstack AND parent_frame == -1
    r_size_in = content_length_in[
        ~df_request_response_in.request_id.isin(callstacks_in.request_id.unique())
        & (df_request_response_in.parent_frame_id == -1)
    ]
    r_size_out = content_length_out[
        ~df_request_response_out.request_id.isin(callstacks_out.request_id.unique())
        & (df_request_response_out.parent_frame_id == -1)
    ]

    features["res_size_rem"] = max(r_size_in.sum() - r_size_out.sum(), 0)

    # 24
    features["n_res_rem"] = max(
        len(df_request_response_in) - len(df_request_response_out), 0
    )

    # 31
    features["r_res_rem"] = (
        max(features["n_res_rem"] / len(df_request_response_in), 0)
        if len(df_request_response_in) > 0
        else 0
    )

    return features


def dom_features(df_dom_in, df_dom_out):
    features = {
        # 6 # of tags and text nodes in initial HTML
        "n_nodes_in": 0,
        # 13 of <iframe> in page
        "n_iframes": 0,
        # 36 % of DOM nodes that are <iframe>
        "r_iframes": 0,
        # 19 Δ in # of sub-documents after blocking
        "n_subdoc_rem": 0,
        # 33 % of DOM nodes that are <html>
        "r_html": 0,
        # 40 % of <html> elements blocked
        "r_html_rem": 0,
        # 31 # of <script> tags in page
        "n_scripts": 0,
    }

    # 6
    features["n_nodes_in"] = len(df_dom_in)

    # 13
    features["n_iframes"] = len(df_dom_in[df_dom_in.nodeName == "iframe"])

    # 36
    features["r_iframes"] = (
        features["n_iframes"] / features["n_nodes_in"]
        if features["n_nodes_in"] > 0
        else 0
    )

    # 19
    features["n_subdoc_rem"] = max(
        features["n_iframes"] - len(df_dom_out[df_dom_out.nodeName == "iframe"]), 0
    )

    # 33
    n_html = len(df_dom_in[df_dom_in.nodeName == "html"])
    features["r_html"] = (
        n_html / features["n_nodes_in"] if features["n_nodes_in"] > 0 else 0
    )

    # 40
    features["r_html_rem"] = (
        max(n_html - len(df_dom_out[df_dom_out.nodeName == "html"]) / n_html, 0)
        if n_html > 0
        else 0
    )

    # 31
    features["n_scripts"] = len(df_dom_in[df_dom_in.nodeName == "script"])

    return features


def graph_features(
    exec_in,
    dom_in,
    requests_in,
):
    # node types: parser, scripts, network requests, dom nodes
    # edge types:
    #  - node insert (parser -> dom)
    #  - resource block: request blocked
    #  - API block: script issues request, request blocked
    #  - dom node parent rel

    features = {
        # 11 # of unique node and edge types
        "n_node_types": 0,
        # 35 # of unique types of actions in entire page
        # "n_edge_type": 0, hard to implement with our setup. not enough context about original implementation
        # 22 # of unique types of actions taken by blocked scripts
        # "n_scr_act_type_rem": 0, hard to implement with our setup. not enough context about original implementation
    }

    n_scripts_in = len(exec_in.script_url.unique())
    n_nodes_in = len(dom_in)
    n_requests_in = len(requests_in.url.unique())

    features["n_node_types"] = n_scripts_in + n_nodes_in + n_requests_in

    return features


def js_dom_features(dom_in, exec_in, dom_out, exec_out):
    features = {
        # 7 # of DOM nodes created by HTML parser prevented by blocking
        "n_dom_html_rem": 0,
        # 12 % of JS DOM nodes created by blocked scripts
        "r_js_scr_rem": 0,
        # 30 # of DOM nodes created by scripts in entire page
        "n_dom_scr": 0,
        # 25 # of DOM nodes created by blocked scripts
        "n_dom_scr_rem": 0,
        # 16 # of DOM node insertions done by blocked scripts
        "n_dom_scr_ins_rem": 0,
        # 21 # of <html> elements created by blocked scripts
        "n_html_scr_rem": 0,
        # 34 % of DOM nodes deletions done by blocked scripts
        "r_dom_scr_del_rem": 0,
        # 27 % of page actions that were network requests
        "r_reqs": 0,
    }

    n_nodes_in = len(dom_in)
    n_nodes_out = len(dom_out)

    n_nodes_js_in = len(exec_in[exec_in.symbol.isin(JS_ALTER_DOM)])
    n_nodes_js_out = len(exec_out[exec_out.symbol.isin(JS_ALTER_DOM)])

    n_nodes_js_insert_in = len(exec_in[exec_in.symbol.isin(JS_INSERT_DOM)])
    n_nodes_js_insert_out = len(exec_out[exec_out.symbol.isin(JS_INSERT_DOM)])

    # 7
    # nodes = nodes_parser + nodes_js
    features["n_dom_html_rem"] = max(
        n_nodes_in - n_nodes_js_in - n_nodes_out + n_nodes_js_out, 0
    )

    # 12
    features["r_js_scr_rem"] = (
        max(n_nodes_js_in - n_nodes_js_out, 0) / n_nodes_js_in
        if n_nodes_js_in > 0
        else 0
    )

    # 30
    features["n_dom_scr"] = n_nodes_js_in

    # 25
    features["n_dom_scr_rem"] = max(n_nodes_js_in - n_nodes_js_out, 0)

    # 16
    features["n_dom_scr_ins_rem"] = max(n_nodes_js_insert_in - n_nodes_js_insert_out, 0)

    n_html_in = len(dom_in[dom_in.nodeName == "html"])
    n_html_out = len(dom_out[dom_out.nodeName == "html"])

    # 21
    features["n_html_scr_rem"] = max(n_html_in - n_html_out, 0)

    # 34
    features["r_dom_scr_del_rem"] = (
        max(n_nodes_js_in - n_nodes_js_out, 0) / n_nodes_js_in
        if n_nodes_js_in > 0
        else 0
    )

    # 27
    n_nodes_js_del_in = len(exec_in[exec_in.symbol.isin(JS_DELETE_DOM)])
    n_nodes_js_del_out = len(exec_out[exec_out.symbol.isin(JS_DELETE_DOM)])

    features["r_reqs"] = (
        (max(n_nodes_js_del_in - n_nodes_js_del_out, 0) / n_nodes_js_del_in)
        if n_nodes_js_del_in > 0
        else 0
    )

    return features


def oth_js_features(exec_in, exec_out):
    features = {
        # 4 of times any script accessed properties on window.navigator
        "n_nav_access": 0,
        # 29 % of window.navigator reads made by blocked scripts
        "r_nav_rem": 0,
        # 14 # of scripts fetched or eval’ed in entire page
        "n_scr_fetch": 0,
        # 5 # of scripts fetched or eval’ed by blocked scripts
        "n_scr_fetch_rem": 0,
        # 8 # of times any script deleted a value from sessionStorage
        "n_ss_del": 0,
        # 18 % of sessionStorage operations done by blocked scripts
        "r_ss_rem": 0,
        # 17 # of document.cookie operations in entire page
        "n_cookie_op": 0,
        # 9 % of document.cookie sets occurring in blocked scripts
        "r_cookie_set_rem": 0,
        # 15 # of times blocked scripts read from document.cookie
        "n_cookie_read_rem": 0,
        # 10 % of localStorage operations occurring in blocked scripts
        "r_ls_rem": 0,
        # 20 # of WebGL calls, over the entire page
        "n_webgl": 0,
        # 23 # of Web API calls made by blocked scripts
        "n_api_rem": 0,
        # 26 % of eventListener removals done by blocked scripts
        "r_event_rem": 0,
        # 28 # of eventListener registrations in entire page
        "n_event": 0,
        # 37 # of window.screen reads over entire page
        "n_screen_read": 0,
        # 38 # of cross-document script-reads in entire page
        "n_xdoc_read": 0,
        # 39 # of localStorage reads over entire page
        "n_ls_read": 0,
    }

    n_nav_in = len(exec_in[exec_in.symbol.isin(JS_NAVIGATOR)])
    n_nav_out = len(exec_out[exec_out.symbol.isin(JS_NAVIGATOR)])

    # 4
    features["n_nav_access"] = n_nav_in

    # 29
    n_nav_get_in = len(
        exec_in[exec_in.symbol.isin(JS_NAVIGATOR) & (exec_in.operation == "get")]
    )
    n_nav_get_out = len(
        exec_out[exec_out.symbol.isin(JS_NAVIGATOR) & (exec_out.operation == "get")]
    )
    features["r_nav_rem"] = (
        max(n_nav_get_in - n_nav_get_out, 0) / n_nav_get_in if n_nav_get_in > 0 else 0
    )

    # 14
    n_fetch_in = len(exec_in[exec_in.symbol.isin(JS_FETCH_OR_EVAL)])
    n_fetch_out = len(exec_out[exec_out.symbol.isin(JS_FETCH_OR_EVAL)])

    features["n_scr_fetch"] = n_fetch_in

    # 5
    features["n_scr_fetch_rem"] = max(n_fetch_in - n_fetch_out, 0)

    # 8
    n_ss_del_in = len(exec_in[exec_in.symbol == "window.sessionStorage.removeItem"])
    n_ss_del_out = len(exec_out[exec_out.symbol == "window.sessionStorage.removeItem"])

    features["n_ss_del"] = n_ss_del_in

    # 18
    features["r_ss_rem"] = (
        max(n_ss_del_in - n_ss_del_out, 0) / n_ss_del_in if n_ss_del_in > 0 else 0
    )

    # 17
    n_cookie_in = len(exec_in[exec_in.symbol.isin(JS_COOKIE)])

    features["n_cookie_op"] = n_cookie_in

    # 9
    n_cookie_set_in = len(
        exec_in[exec_in.symbol.isin(JS_COOKIE) & (exec_in.operation == "set")]
    )

    n_cookie_set_out = len(
        exec_out[exec_out.symbol.isin(JS_COOKIE) & (exec_out.operation == "set")]
    )

    features["r_cookie_set_rem"] = (
        max(n_cookie_set_in - n_cookie_set_out, 0) / n_cookie_set_in
        if n_cookie_set_in > 0
        else 0
    )

    # 15
    n_cookie_read_in = len(
        exec_in[exec_in.symbol.isin(JS_COOKIE) & (exec_in.operation == "get")]
    )

    n_cookie_read_out = len(
        exec_out[exec_out.symbol.isin(JS_COOKIE) & (exec_out.operation == "get")]
    )

    features["n_cookie_read_rem"] = max(n_cookie_read_in - n_cookie_read_out, 0)

    # 10
    n_ls_in = len(
        exec_in[
            exec_in.symbol.isin(JS_LOCAL_STORAGE)
            & ((exec_in.operation == "get") | (exec_in.operation == "set"))
        ]
    )

    n_ls_out = len(
        exec_out[
            exec_out.symbol.isin(JS_LOCAL_STORAGE)
            & ((exec_out.operation == "get") | (exec_out.operation == "set"))
        ]
    )

    features["r_ls_rem"] = max(n_ls_in - n_ls_out, 0) / n_ls_in if n_ls_in > 0 else 0

    # 20
    n_webgl_in = len(exec_in[exec_in.symbol.isin(JS_WEBGL)])

    features["n_webgl"] = n_webgl_in

    # 23
    n_api_in = len(exec_in[exec_in.symbol.isin(JS_API)])
    n_api_out = len(exec_out[exec_out.symbol.isin(JS_API)])

    features["n_api_rem"] = max(n_api_in - n_api_out, 0)

    # 26
    n_event_rem_in = len(exec_in[exec_in.symbol == "window.removeEventListener"])
    n_event_rem_out = len(exec_out[exec_out.symbol == "window.removeEventListener"])

    features["r_event_rem"] = (
        max(n_event_rem_in - n_event_rem_out, 0) / n_event_rem_in
        if n_event_rem_in > 0
        else 0
    )

    # 28
    n_event_in = len(exec_in[exec_in.symbol == "window.addEventListener"])

    features["n_event"] = n_event_in

    # 37
    n_screen_read_in = len(exec_in[exec_in.symbol.isin(JS_SCREEN)])

    features["n_screen_read"] = n_screen_read_in

    # 38
    n_xdoc_read_in = len(exec_in[exec_in.symbol == "window.postMessage"])

    features["n_xdoc_read"] = n_xdoc_read_in

    # 39
    n_ls_read_in = len(
        exec_in[exec_in.symbol.isin(JS_LOCAL_STORAGE) & (exec_in.operation == "get")]
    )

    features["n_ls_read"] = n_ls_read_in

    return features


def process_visit_pair(
    visit_id_in,
    visit_id_out,
    dom_in,
    req_in,
    dom_out,
    req_out,
    database: Database,
):
    JS_NODES = (
        JS_NAVIGATOR
        + JS_ALTER_DOM
        + JS_INSERT_DOM
        + JS_DELETE_DOM
        + JS_FETCH_OR_EVAL
        + JS_SESSION_STORAGE
        + JS_COOKIE
        + JS_LOCAL_STORAGE
        + JS_WEBGL
        + JS_API
        + JS_EVENT
        + JS_SCREEN
    )

    callstacks_in = database.get_callstacks(visit_id_in)
    callstacks_out = database.get_callstacks(visit_id_out)
    js_in = database.get_all_javascript_events([visit_id_in], symbols=JS_NODES)
    js_out = database.get_all_javascript_events([visit_id_out], symbols=JS_NODES)

    features = {}

    features |= req_features(req_in, callstacks_in, req_out, callstacks_out)
    features |= dom_features(dom_in, dom_out)
    features |= graph_features(js_in, dom_in, req_in)
    features |= js_dom_features(dom_in, js_in, dom_out, js_out)
    features |= oth_js_features(js_in, js_out)

    return features


def extract_features(crawl_dir: Path, output_dir: Path, issues: list = None, label=1):
    db_file: Path = crawl_dir / "crawl-data.sqlite"
    exp_file: Path = crawl_dir / "experiments.csv"

    output_dir.mkdir(parents=True, exist_ok=True)

    with Database(db_file, exp_file) as database, DataframeCSVStorageController(
        output_dir,
        [f"adgraph_features_{label}.csv"],
    ) as storage:
        sites_visits = database.sites_visits()

        if issues is not None:
            sites_visits = sites_visits[(sites_visits.issue_id.isin(issues))]

        visit_ids = list(
            set(sites_visits.visit_id_a.unique())
            | set(sites_visits.visit_id_b.unique())
            | set(sites_visits.visit_id_u.unique())
        )

        df_dom = database.get_dom(visit_ids)
        df_request_response = database.get_request_response(visit_ids)

        if len(sites_visits) == 0:
            raise GraphExtractionError

        # after to before -> broken
        # no to after -> not broken
        # before to after -> not broken
        # no to before -> broken

        visit_pairs = [
            ("visit_id_a", "visit_id_b", label)
            # ("visit_id_u", "visit_id_a", -1),
            # ("visit_id_b", "visit_id_a", -1),
            # ("visit_id_u", "visit_id_b", 1),
        ]

        features = []

        for _, issue in tqdm(sites_visits.iterrows(), total=len(sites_visits)):
            for visit_id_a, visit_id_b, label in visit_pairs:
                _features = process_visit_pair(
                    issue[visit_id_a],
                    issue[visit_id_b],
                    df_dom[df_dom.visit_id == issue[visit_id_a]],
                    df_request_response[
                        df_request_response.visit_id == issue[visit_id_a]
                    ],
                    df_dom[df_dom.visit_id == issue[visit_id_b]],
                    df_request_response[
                        df_request_response.visit_id == issue[visit_id_b]
                    ],
                    database,
                )

                _features["visit_id_prev"] = issue[visit_id_a]
                _features["visit_id_new"] = issue[visit_id_b]
                _features["issue_id"] = issue.issue_id
                _features["is_broken"] = label

                features.append(_features)

                storage.save(
                    f"adgraph_features_{label}.csv",
                    pd.DataFrame([_features]),
                )

    return pd.DataFrame(features)


def extract_features_mult(crawl_dirs: list, output_dir: Path):
    features = []

    for crawl_dir, label in crawl_dirs:
        features.append(extract_features(crawl_dir, output_dir, label=label))

    # align columns
    for i in range(1, len(features)):
        features[i] = features[i][features[0].columns]

    features = pd.concat(features)
    features.to_csv(output_dir / "adgraph_features.csv", index=False)

    return features


FEATURES = [
    "n_dom_html_rem",
    "r_js_scr_rem",
    "n_dom_scr",
    "n_dom_scr_rem",
    "n_dom_scr_ins_rem",
    "n_html_scr_rem",
    "r_dom_scr_del_rem",
    "r_reqs",
    "n_nav_access",
    "r_nav_rem",
    "n_scr_fetch",
    "n_scr_fetch_rem",
    "n_ss_del",
    "r_ss_rem",
    "n_cookie_op",
    "r_cookie_set_rem",
    "n_cookie_read_rem",
    "r_ls_rem",
    "n_webgl",
    "n_api_rem",
    "r_event_rem",
    "n_event",
    "n_screen_read",
    "n_xdoc_read",
    "n_ls_read",
    "r_reqs_subdoc_rem",
    "req_bytes_sent_rem",
    "res_size_rem",
    "n_res_rem",
    "r_res_rem",
    "n_nodes_in",
    "n_iframes",
    "r_iframes",
    "n_subdoc_rem",
    "r_html",
    "r_html_rem",
    "n_scripts",
]


def split_train_test(df, features, random_state=None):
    X = df[features].values
    y = df["is_broken"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    return X_train, X_test, y_train, y_test


class AdgraphClassifier(ModelPipeline):
    def __init__(self, model, label_encoder=LabelEncoder, scaler=None):
        super().__init__(
            model, Preprocessor(FEATURES), label_encoder, scaler, "is_broken"
        )

    def eval(self, df, resampler=SMOTE, random_state=None):
        return super().eval(df, split_train_test, resampler, random_state, False)


class VisitPairModelWrapper:
    def __init__(self, model: AdgraphClassifier, features: list):
        self.model = model
        self.features = features

    def predict(self, visit_prev, visit_new):
        features = self.features[
            (self.features["visit_id_prev"] == visit_prev)
            & (self.features["visit_id_new"] == visit_new)
        ]

        return self.model.predict(features)[0]
    
class IssueModelWrapper:
    def __init__(self, model: AdgraphClassifier, features: list):
        self.model = model
        self.features = features

    def predict(self, issue):
        features = self.features[
            (self.features.issue_id == issue)
        ]

        return self.model.predict(features)[0]


# if __name__ == "__main__":
#     ISSUES = [
#         52400,
#         53389,
#         51952,
#         52109,
#         143915,
#         157859,
#         157392,
#         155285,
#         52408,
#         52495,
#         161198,
#         142957,
#         151923,
#         142668,
#         149809,
#         53452,
#         143215,
#         139965,
#         155130,
#         149056,
#         158902,
#         139041,
#         143708,
#         52041,
#         153312,
#         149568,
#         159565,
#         157220,
#         143339,
#         138888,
#         162559,
#         143247,
#         160460,
#         151732,
#         51628,
#         163156,
#         153946,
#         140162,
#         141652,
#         155639,
#         157550,
#         161355,
#         155384,
#         140726,
#         144631,
#         151633,
#         156996,
#         157756,
#         154643,
#         155784,
#         151582,
#         151910,
#         151088,
#         53401,
#         161751,
#         139622,
#         161344,
#         142191,
#         144101,
#         53591,
#         159646,
#         51641,
#         151169,
#         52979,
#         138802,
#         155695,
#         138932,
#         158090,
#         161236,
#         159143,
#         152847,
#         149935,
#         161962,
#         139376,
#         149019,
#         52181,
#         53059,
#         160241,
#         139618,
#         152478,
#         155634,
#         144923,
#         160568,
#         152957,
#         141090,
#         158134,
#         158238,
#         52281,
#         155206,
#         152530,
#         156883,
#         150915,
#         155538,
#         140831,
#         149561,
#         51938,
#         141434,
#         151987,
#         152234,
#         154709,
#         52836,
#         154814,
#         155363,
#         154623,
#         150114,
#         150448,
#         144074,
#         149892,
#         52046,
#         51951,
#         160679,
#         142365,
#         152020,
#         158664,
#         151618,
#         52949,
#         140301,
#         156209,
#         161237,
#         159894,
#         161056,
#         160995,
#         141163,
#         52081,
#         151060,
#         53298,
#         52546,
#         138555,
#         154556,
#         162790,
#         52583,
#         151824,
#         143693,
#         161189,
#         162304,
#         163109,
#         162109,
#         144838,
#         158904,
#         155927,
#         143184,
#         151059,
#         150501,
#         159100,
#         53445,
#         156072,
#         161447,
#         154601,
#         144648,
#         157284,
#         52173,
#         153959,
#         150486,
#         152800,
#         151483,
#         162943,
#         139866,
#         161389,
#         159355,
#         160156,
#         139155,
#         156646,
#         153176,
#         152845,
#         52832,
#         52589,
#         151706,
#         151272,
#         140588,
#         53715,
#         149439,
#         138813,
#         158921,
#         153267,
#         151887,
#         151296,
#         142547,
#         153746,
#         143139,
#         155189,
#         161051,
#         143407,
#         151186,
#         156723,
#         154274,
#         144192,
#         141425,
#         52267,
#         159792,
#         150628,
#         140777,
#         156752,
#         161361,
#         140710,
#         153380,
#         142100,
#         52356,
#         144474,
#         156179,
#         53427,
#         152815,
#         157773,
#         155555,
#         158401,
#         150082,
#         161517,
#         149131,
#         138742,
#         52051,
#         148866,
#         159441,
#         149358,
#         159941,
#         151369,
#         152108,
#         154970,
#         159533,
#         153233,
#         161741,
#         161886,
#         148691,
#         153062,
#         157246,
#         156796,
#         157766,
#         53346,
#         157906,
#         151053,
#         154859,
#         159998,
#         159857,
#         140718,
#         142606,
#         161931,
#         143252,
#         151031,
#         161710,
#         139450,
#         157743,
#         142489,
#         162039,
#         151568,
#         142369,
#         156644,
#         144676,
#         150596,
#         154493,
#         149177,
#         157890,
#         150856,
#         161645,
#         151688,
#         162977,
#         155677,
#         151030,
#         139159,
#         158932,
#         143282,
#         161722,
#         139471,
#         162167,
#         161279,
#         150480,
#         149022,
#         153646,
#         52776,
#         52485,
#         139712,
#         53299,
#         142773,
#         160735,
#         162102,
#         53202,
#         153277,
#         52135,
#         156867,
#         162750,
#         143102,
#         150407,
#         141077,
#         152325,
#         139578,
#         155087,
#         153174,
#         51708,
#         142514,
#         139046,
#         144173,
#         162358,
#         144552,
#         160401,
#         140690,
#         52315,
#         53038,
#         157455,
#         157533,
#         154714,
#         149042,
#         163150,
#         153200,
#         158667,
#         140891,
#         158198,
#         150328,
#         154023,
#         144897,
#         154057,
#         149466,
#         139004,
#         52160,
#         160360,
#         141809,
#         161841,
#         53468,
#         142440,
#         151405,
#         160901,
#         143484,
#         140781,
#         153318,
#         143486,
#         139943,
#         159864,
#         139529,
#         139980,
#         52930,
#         153124,
#         142900,
#         143409,
#         143240,
#         151468,
#         139163,
#         150115,
#         153458,
#         154277,
#         144064,
#         139968,
#         142724,
#         155722,
#         158945,
#         52331,
#         140178,
#         52669,
#         158202,
#         52419,
#         153513,
#         52222,
#         141319,
#         157694,
#         149310,
#         52869,
#         161801,
#         144884,
#         161802,
#         51629,
#         53359,
#         160891,
#         52579,
#         152966,
#         53159,
#         161750,
#         152356,
#         139572,
#         153420,
#         150415,
#         138803,
#         157544,
#         149488,
#         153239,
#         159972,
#         150867,
#         161338,
#         149138,
#         161018,
#         154050,
#         144097,
#         156551,
#         52990,
#         149284,
#         139063,
#         153810,
#         142845,
#         53373,
#         155565,
#         51911,
#         152186,
#         141710,
#         153554,
#         152338,
#         152030,
#         152279,
#         149414,
#         150859,
#         150409,
#         160468,
#         51865,
#         152319,
#         51737,
#         52039,
#         160124,
#         52753,
#         52925,
#         140523,
#         141024,
#         157184,
#         144911,
#         140903,
#         156728,
#         160554,
#         154432,
#         162001,
#         142816,
#         144144,
#         157140,
#         155638,
#         149899,
#         143582,
#         51861,
#         53358,
#         153514,
#         162117,
#         140059,
#         143756,
#         149495,
#         149769,
#         154488,
#         141540,
#         159060,
#         159130,
#         162953,
#         156403,
#         142788,
#         53752,
#         150868,
#         154061,
#         162780,
#         151874,
#         149336,
#         160154,
#         143850,
#         52345,
#         162483,
#         51868,
#         51619,
#         52812,
#         144199,
#         140783,
#         151951,
#         161318,
#         144042,
#         153647,
#         153326,
#         140576,
#         150369,
#         149948,
#         162827,
#         156229,
#         152892,
#         152820,
#         159077,
#         161486,
#         152346,
#         160751,
#         158498,
#         157010,
#         142236,
#         160684,
#         138835,
#         142312,
#         150590,
#         144540,
#         53486,
#         51862,
#         52615,
#         141721,
#         52918,
#         153698,
#         138973,
#         53190,
#         159304,
#         152339,
#         157934,
#         153734,
#         158247,
#         143853,
#         149366,
#         161885,
#         143472,
#         140803,
#         152349,
#         153936,
#         162524,
#         51890,
#         149927,
#         52411,
#         142758,
#         151017,
#         143493,
#         52539,
#     ]

#     __dir__ = Path(__file__).parent


#     extract_features(
#         __dir__ / "../crawl-out/adguard-full",
#         __dir__ / "data",
#         issues=ISSUES
#     )
