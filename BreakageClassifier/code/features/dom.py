from multiprocessing import Process, Queue, current_process
from typing import Tuple
import pandas as pd
import json
from difflib import SequenceMatcher
from BreakageClassifier.code.features.utils import _parse_attrs

from BreakageClassifier.code.utils import graph_node_id

pd.options.mode.chained_assignment = None


def parse_attributes(row: pd.Series):
    attrs = _parse_attrs(row)

    # vis = json.loads(row.visual_cues)

    # the attrs we care about are id, class, type, (src or href)
    return (
        attrs.get("id", None),
        attrs.get("class", None),
        attrs.get("src", attrs.get("href", None)),
    )


def score_similarity(left_node: pd.Series, right_node: pd.Series) -> float:
    """scores the similarity between two html nodes. the score is an indicator of the likelihood that left_node got edited to right_node.

    Args:
        left_node (pd.Series): _description_
        right_node (pd.Series): _description_

    Returns:
        float: _description_
    """

    left_attrs = _parse_attrs(left_node)
    right_attrs = _parse_attrs(right_node)

    if (
        "attr_id" in left_node.index
        and isinstance(left_node.attr_id, str)
        and left_node.attr_id == right_node.attr_id
        or "attr_src" in left_node.index
        and isinstance(left_node.attr_src, str)
        and left_node.attr_src == right_node.attr_src
    ):
        return 1

    # checking that ids are different
    if (
        "attr_id" in left_node.index
        and "attr_id" in right_node.index
        and isinstance(left_node.attr_id, str)
        and left_node.attr_id != right_node.attr_id
        and (
            "attr_src" not in left_node.index
            or "attr_src" not in right_node.index
            or left_node.attr_src != right_node.attr_src
        )
        and (
            "name" not in left_attrs
            or "name" not in right_attrs
            or left_attrs["name"] != right_attrs["name"]
        )
    ):
        return 0

    # check that one element has source while other has not
    if ("attr_src" not in left_node.index) ^ ("attr_src" not in right_node.index):
        return 0

    # computing the score
    weights = {"class": 0.6, "src": 0.3, "text": 0.4, "attributes": 0.5, "visual": 0.2}

    if "attr_class" not in left_node.index and "attr_class" not in right_node.index:
        weights["class"] = 0
    if "attr_src" not in left_node.index and "attr_src" not in right_node.index:
        weights["src"] = 0

    # check attribute differences
    left_attr_keys = set(
        [
            (x["key"], x["value"])
            for x in json.loads(left_node.attributes)
            if x["key"] not in ["class", "id", "src"]
        ]
    )
    right_attr_keys = set(
        [
            (x["key"], x["value"])
            for x in json.loads(left_node.attributes)
            if x["key"] not in ["class", "id", "src"]
        ]
    )

    score_attr = 0
    if len(left_attr_keys) != 0 or len(right_attr_keys) != 0:
        sim_attr_keys = left_attr_keys.intersection(right_attr_keys)
        score_attr = len(sim_attr_keys) / min(len(left_attr_keys), len(right_attr_keys))

    else:
        weights["attributes"] = 0

    # checking class
    score_class = 0
    if (
        "attr_class" in left_node.index
        and "attr_class" in right_node.index
        and isinstance(left_node.attr_class, str)
        and isinstance(right_node.attr_class, str)
    ):
        left_classes = set(left_node.attr_class.split(" "))
        right_classes = set(right_node.attr_class.split(" "))
        inter_classes = left_classes.intersection(right_classes)

        score_class = len(inter_classes) / min(len(left_classes), len(right_classes))
    elif (
        "attr_class" not in left_node.index and "attr_class" not in right_node.index
    ) or (
        not isinstance(left_node.attr_class, str)
        and not isinstance(right_node.attr_class, str)
    ):
        weights["class"] = 0
    else:
        return 0

    # checking source
    score_src = 0
    if (
        "attr_src" in left_node.index
        and "attr_src" in right_node.index
        and isinstance(left_node.attr_src, str)
        and isinstance(right_node.attr_src, str)
    ):
        match = SequenceMatcher(
            None, left_node.attr_src, right_node.attr_src
        ).find_longest_match()

        try:
            score_src = match.size / min(
                len(left_node.attr_src), len(right_node.attr_src)
            )
        except ZeroDivisionError:
            score_src = 0

    elif ("attr_src" not in left_node.index and "attr_src" not in right_node.index) or (
        not isinstance(left_node.attr_src, str)
        and not isinstance(right_node.attr_src, str)
    ):
        weights["src"] = 0
    else:
        return 0

    # checking text
    score_text = 0
    if (
        "visual_cues" in left_node.index
        and "visual_cues" in right_node.index
        and isinstance(left_node.visual_cues, str)
        and isinstance(right_node.visual_cues, str)
        and "text" in left_node.visual_cues
        and "text" in right_node.visual_cues
    ):
        left_text = json.loads(left_node.visual_cues)["text"]
        right_text = json.loads(right_node.visual_cues)["text"]

        left_words = set([x.lower().strip() for x in left_text.split(" ")])
        right_words = set([x.lower().strip() for x in right_text.split(" ")])

        # print(left_words)

        # match = SequenceMatcher(
        #     None, left_text, right_text
        # ).find_longest_match()
        # score_text = match.size / min(
        #     len(left_text), len(right_text)
        # )

        score_text = len(left_words.intersection(right_words)) / min(
            len(left_words), len(right_words)
        )

    elif (
        "visual_cues" not in left_node.index and "visual_cues" not in right_node.index
    ) or (
        not isinstance(left_node.visual_cues, str)
        and not isinstance(right_node.visual_cues, str)
    ):
        weights["text"] = 0
    else:
        return 0

    # checking visual
    score_visual = 0
    if (
        "visual_cues" in left_node.index
        and "visual_cues" in right_node.index
        and isinstance(left_node.visual_cues, str)
        and isinstance(right_node.visual_cues, str)
    ):
        left_cues = json.loads(left_node.visual_cues)
        right_cues = json.loads(right_node.visual_cues)

        try:
            d = (
                (left_cues["bounds"]["x"] - right_cues["bounds"]["x"]) ** 2
                + (left_cues["bounds"]["y"] - right_cues["bounds"]["y"]) ** 2
            ) / (
                max(left_cues["bounds"]["x"], right_cues["bounds"]["x"]) ** 2
                + max(left_cues["bounds"]["y"], right_cues["bounds"]["y"]) ** 2
            )
        except ZeroDivisionError:
            d = 0

        try:
            dw = abs(
                left_cues["bounds"]["width"] - right_cues["bounds"]["width"]
            ) / max(left_cues["bounds"]["width"], right_cues["bounds"]["width"])
        except ZeroDivisionError:
            dw = 0

        try:
            dh = abs(
                left_cues["bounds"]["height"] - right_cues["bounds"]["height"]
            ) / max(left_cues["bounds"]["height"], right_cues["bounds"]["height"])
        except ZeroDivisionError:
            dh = 0

        score_visual = 1 - (d**0.5 + dw + dh) / 3

        score_visual = score_visual if score_visual < 0.95 else 1

    elif (
        "visual_cues" not in left_node.index and "visual_cues" not in right_node.index
    ) or (
        not isinstance(left_node.visual_cues, str)
        and not isinstance(right_node.visual_cues, str)
    ):
        weights["visual"] = 0
    else:
        return 0

    # print(score_class, score_src, score_text, score_attr, score_visual)
    # print(weights)

    score = (
        score_class * weights["class"]
        + score_src * weights["src"]
        + score_text * weights["text"]
        + score_attr * weights["attributes"]
        + score_visual * weights["visual"]
    ) / sum(weights.values())

    return score


def add_saliencies_to_tree(
    dom: pd.DataFrame, salient_nodes: pd.DataFrame
) -> pd.DataFrame:
    for _, salient_node in salient_nodes.iterrows():
        candidates = dom[(dom["nodeName"] == salient_node.nodeName)]

        if not candidates.empty:
            candidates["score"] = candidates.apply(
                lambda child: score_similarity(child, salient_node), axis=1
            )

            candidate: pd.Series = candidates[
                candidates["score"] == max(candidates["score"])
            ].iloc[0]

            max_score = candidate.score

            if 0.75 < max_score <= 1:
                # the candidate is true

                dom.loc[candidate.name]["saliency"] = 1.0
                dom.loc[
                    (dom["t_enter"] > candidate.t_enter)
                    & (dom["t_leave"] < candidate.t_leave),
                    "saliency",
                ] = 1.0

    return dom


def clean_tree(tree_df: pd.DataFrame) -> pd.DataFrame:
    nodes_id_drop = set()
    nodes_index_drop = list()

    # fix node id
    tree_df["id"] = tree_df["id"].astype(int)

    for i, node in tree_df.iterrows():
        if node.parent_id == -1:
            continue

        parent = tree_df[tree_df["id"] == node.parent_id].iloc[0]

        if _skip_node_descendents(parent) or parent.id in nodes_id_drop:
            nodes_id_drop.add(node.id)
            nodes_index_drop.append(i)
    return tree_df.drop(index=nodes_index_drop)


def _skip_node_descendents(node: pd.Series) -> pd.DataFrame:
    if node.nodeName in ["svg"]:
        return True

    return False


def compare_trees(
    left_tree: pd.DataFrame, right_tree: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """generates a common subtree and an edit group subtrees.

    Returns:
        pd.DataFrame, pd.DataFrame: the first dataframe is the common subtree and the second one is the edits dataframe
    """
    if (
        left_tree[left_tree["parent_id"] == -1].empty
        or right_tree[right_tree["parent_id"] == -1].empty
    ):
        return pd.DataFrame(), pd.DataFrame()

    left_root = left_tree[left_tree["parent_id"] == -1].iloc[0].id
    right_root = right_tree[right_tree["parent_id"] == -1].iloc[0].id

    def _mark_sub_tree_altered(
        root_id, tree: pd.DataFrame, edited=False, rem_in_l=False, rem_in_r=False
    ):
        root: pd.Series = tree[tree["id"] == root_id].iloc[0]
        subtree: pd.DataFrame = tree[
            (tree["t_enter"] > root.t_enter) & (tree["t_leave"] < root.t_leave)
        ].copy(deep=True)

        subtree["edited"] = edited
        subtree["rem_in_l"] = rem_in_l
        subtree["rem_in_r"] = rem_in_r
        subtree["is_root"] = False

        return subtree

    def _compare_trees_rec(
        left_root_id, right_root_id, is_root=True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # the children of the left root
        left_children: pd.DataFrame = left_tree[
            left_tree["parent_id"] == left_root_id
        ].copy(deep=True)

        # the children of the right root
        right_children: pd.DataFrame = right_tree[
            right_tree["parent_id"] == right_root_id
        ].copy(deep=True)

        right_children["processed"] = False

        # setup the nodes to query

        if not left_children.empty:
            left_children = left_children[
                (left_children.apply(lambda x: "openwpm" not in x.attributes, axis=1))
            ]

        if not left_children.empty:
            left_children[["attr_id", "attr_class", "attr_src"]] = left_children.apply(
                parse_attributes, axis=1, result_type="expand"
            )

        if not right_children.empty:
            right_children = right_children[
                (right_children.apply(lambda x: "openwpm" not in x.attributes, axis=1))
            ]

        if not right_children.empty:
            right_children[
                ["attr_id", "attr_class", "attr_src"]
            ] = right_children.apply(parse_attributes, axis=1, result_type="expand")

        common_children = pd.DataFrame(columns=left_children.columns)
        altered_children = pd.DataFrame(
            columns=[
                "other_attr_class",
                "other_attr_id",
                "other_attr_src",
                "other_attributes",
                "other_browser_id",
                "other_id",
                "other_parent_id",
                "other_t_enter",
                "other_t_leave",
                "other_visit_id",
                "other_visual_cues",
                "processed",
                "is_root",
            ]
        )

        for _, left_child in left_children.iterrows():
            # don't care about openwpm instrumentation
            if "openwpm" in left_child.attributes:
                continue

            # get all the elements from the same tag type of other tree
            candidates = right_children[
                (right_children["nodeName"] == left_child.nodeName)
                & (~right_children["processed"])
            ]

            if not candidates.empty:
                candidates["score"] = candidates.apply(
                    lambda child: score_similarity(child, left_child), axis=1
                )

                candidate: pd.Series = candidates[
                    candidates["score"] == max(candidates["score"])
                ].iloc[0]

                max_score = candidate.score

            else:
                max_score = 0

            _left = left_child.copy(deep=True)

            if max_score < 0.75:
                # the element must have been removed
                _left["edited"] = False
                _left["rem_in_r"] = True
                _left["rem_in_l"] = False
                _left["is_root"] = is_root
                altered_children = altered_children.append(_left)

            elif max_score < 1:
                # the element was altered

                _left["edited"] = True
                _left["rem_in_r"] = False
                _left["rem_in_l"] = False
                _left["other_browser_id"] = candidate.browser_id
                _left["other_visit_id"] = candidate.visit_id
                _left["other_id"] = candidate.id
                _left["other_attributes"] = candidate.attributes
                _left["other_visual_cues"] = candidate.visual_cues
                _left["other_parent_id"] = candidate.parent_id
                _left["other_attr_id"] = candidate.attr_id
                _left["other_attr_class"] = candidate.attr_class
                _left["other_attr_src"] = candidate.attr_src
                _left["other_t_enter"] = candidate.t_enter
                _left["other_t_leave"] = candidate.t_leave
                _left["is_root"] = is_root
                altered_children = altered_children.append(_left)

                # we need to remove elements from the pool of choices
                right_children.loc[
                    right_children["id"] == candidate.id, "processed"
                ] = True

            else:
                # the element is the same
                candidate["right_id"] = candidate.id
                candidate["left_id"] = left_child.id
                common_children = common_children.append(
                    candidate.drop(columns=["score", "right", "browser_id", "visit_id"])
                )

                # we need to remove elements from the pool of choices
                right_children.loc[
                    right_children["id"] == candidate.id, "processed"
                ] = True

        # the remaining unprocessed nodes in right must be removed from left
        altered_children_r = right_children[~right_children["processed"]]
        if not altered_children_r.empty:
            altered_children_r["edited"] = False
            altered_children_r["rem_in_l"] = True
            altered_children_r["rem_in_r"] = False
            altered_children_r["is_root"] = is_root
            altered_children = pd.concat([altered_children, altered_children_r], axis=0)

        # recursive check common tree

        common_descendents = pd.DataFrame()
        altered_descendents = pd.DataFrame()

        # TODO: parallelize

        def __handle_common_child(common_child):
            _, common_child = common_child

            return _compare_trees_rec(
                common_child.left_id, common_child.right_id, is_root=is_root
            )

        if not common_children.empty:
            _c = []
            _a = []

            for row in common_children.iterrows():
                _c_child, _a_child = __handle_common_child(row)
                _c.append(_c_child)
                _a.append(_a_child)

            if len(_c):
                common_descendents = pd.concat(_c)
            if len(_a):
                altered_descendents = pd.concat(_a)

        # recursive add altered subtrees
        for _, altered_child in altered_children.iterrows():
            if altered_child.rem_in_l:
                _a = _mark_sub_tree_altered(altered_child.id, right_tree, rem_in_l=True)
                altered_descendents = pd.concat([altered_descendents, _a], axis=0)

            elif altered_child.rem_in_r:
                _a = _mark_sub_tree_altered(altered_child.id, left_tree, rem_in_r=True)
                altered_descendents = pd.concat([altered_descendents, _a], axis=0)

            else:
                _c, _a = _compare_trees_rec(
                    altered_child.id, altered_child.other_id, is_root=False
                )

                altered_descendents = pd.concat([altered_descendents, _a], axis=0)

                if not _c.empty:
                    _c.drop(
                        columns=["processed", "score", "left_id", "right_id"],
                        inplace=True,
                    )

                    _c["is_root"] = False

                    altered_descendents = pd.concat([altered_descendents, _c], axis=0)

        altered_children = pd.concat([altered_children, altered_descendents], axis=0)
        common_children = pd.concat([common_children, common_descendents], axis=0)

        if not altered_children.empty:
            altered_children = altered_children.drop(columns=["processed"])

        return common_children, altered_children

    return _compare_trees_rec(left_root, right_root)


def compare_trees_parallel(
    left_tree: pd.DataFrame, right_tree: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """generates a common subtree and an edit group subtrees.

    Returns:
        pd.DataFrame, pd.DataFrame: the first dataframe is the common subtree and the second one is the edits dataframe
    """

    if (
        left_tree[left_tree["parent_id"] == -1].empty
        or right_tree[right_tree["parent_id"] == -1].empty
    ):
        return pd.DataFrame(), pd.DataFrame()

    left_root = left_tree[left_tree["parent_id"] == -1].iloc[0].id
    right_root = right_tree[right_tree["parent_id"] == -1].iloc[0].id

    def _mark_sub_tree_altered(
        root_id, tree: pd.DataFrame, edited=False, rem_in_l=False, rem_in_r=False
    ):
        root: pd.Series = tree[tree["id"] == root_id].iloc[0]
        subtree: pd.DataFrame = tree[
            (tree["t_enter"] > root.t_enter) & (tree["t_leave"] < root.t_leave)
        ].copy(deep=True)

        subtree["edited"] = edited
        subtree["rem_in_l"] = rem_in_l
        subtree["rem_in_r"] = rem_in_r
        subtree["is_root"] = False
        # subtree[['attr_id', 'attr_class', 'attr_src']] = subtree.apply(parse_attributes, axis=1, result_type='expand')

        return subtree

    def _compare_trees_rec(
        left_root_id,
        right_root_id,
        is_root,
        edited_queue: Queue = None,
        common_queue: Queue = None,
        awaited_tasks: Queue = None,
        parent_pid=None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # define containers
        _awaited_tasklist = []

        # the children of the left root
        left_children: pd.DataFrame = left_tree[
            left_tree["parent_id"] == left_root_id
        ].copy(deep=True)

        # the children of the right root
        right_children: pd.DataFrame = right_tree[
            right_tree["parent_id"] == right_root_id
        ].copy(deep=True)

        right_children["processed"] = False

        # setup the nodes to query

        if not left_children.empty:
            left_children = left_children[
                (left_children.apply(lambda x: "openwpm" not in x.attributes, axis=1))
            ]

        # if not left_children.empty:
        #     left_children[["attr_id", "attr_class", "attr_src"]] = left_children.apply(
        #         parse_attributes, axis=1, result_type="expand"
        #     )

        if not right_children.empty:
            right_children = right_children[
                (right_children.apply(lambda x: "openwpm" not in x.attributes, axis=1))
            ]

        # if not right_children.empty:
        #     right_children[
        #         ["attr_id", "attr_class", "attr_src"]
        #     ] = right_children.apply(parse_attributes, axis=1, result_type="expand")

        common_children = pd.DataFrame(columns=left_children.columns)
        altered_children = pd.DataFrame(
            columns=[
                "other_attr_class",
                "other_attr_id",
                "other_attr_src",
                "other_attributes",
                "other_browser_id",
                "other_id",
                "other_parent_id",
                "other_t_enter",
                "other_t_leave",
                "other_visit_id",
                "other_visual_cues",
                "saliency",
                "party",
                "processed",
                "is_root",
            ]
        )

        for _, left_child in left_children.iterrows():
            # don't care about openwpm instrumentation
            if "openwpm" in left_child.attributes:
                continue

            # get all the elements from the same tag type of other tree
            candidates = right_children[
                (right_children["nodeName"] == left_child.nodeName)
                & (~right_children["processed"])
            ]

            if not candidates.empty:
                candidates["score"] = candidates.apply(
                    lambda child: score_similarity(child, left_child), axis=1
                )

                candidate: pd.Series = candidates[
                    candidates["score"] == max(candidates["score"])
                ].iloc[0]

                max_score = candidate.score

            else:
                max_score = 0

            _left = left_child.copy(deep=True)

            if max_score < 0.6:
                # the element must have been removed
                _left["edited"] = False
                _left["rem_in_r"] = True
                _left["rem_in_l"] = False
                _left["is_root"] = is_root
                altered_children = pd.concat([altered_children, pd.DataFrame([_left])])

            elif max_score < 1:
                # the element was altered

                _left["edited"] = True
                _left["rem_in_r"] = False
                _left["rem_in_l"] = False
                _left["other_browser_id"] = candidate.browser_id
                _left["other_visit_id"] = candidate.visit_id
                _left["other_id"] = candidate.id
                _left["other_attributes"] = candidate.attributes
                _left["other_visual_cues"] = candidate.visual_cues
                _left["other_parent_id"] = candidate.parent_id
                _left["other_attr_id"] = candidate.attr_id
                _left["other_attr_class"] = candidate.attr_class
                _left["other_attr_src"] = candidate.attr_src
                _left["other_t_enter"] = candidate.t_enter
                _left["other_t_leave"] = candidate.t_leave
                _left["is_root"] = is_root

                altered_children = pd.concat([altered_children, pd.DataFrame([_left])])

                # we need to remove elements from the pool of choices
                right_children.loc[
                    right_children["id"] == candidate.id, "processed"
                ] = True

            else:
                # the element is the same
                candidate["right_id"] = candidate.id
                candidate["left_id"] = left_child.id
                common_children = pd.concat(
                    [
                        common_children,
                        pd.DataFrame(
                            [
                                candidate.drop(
                                    columns=["score", "right", "browser_id", "visit_id"]
                                )
                            ]
                        ),
                    ]
                )

                # we need to remove elements from the pool of choices
                right_children.loc[
                    right_children["id"] == candidate.id, "processed"
                ] = True

        # the remaining unprocessed nodes in right must be removed from left
        altered_children_r = right_children[~right_children["processed"]]
        if not altered_children_r.empty:
            altered_children_r["edited"] = False
            altered_children_r["rem_in_l"] = True
            altered_children_r["rem_in_r"] = False
            altered_children_r["is_root"] = is_root

            altered_children = pd.concat([altered_children, altered_children_r], axis=0)

        # recursive check common tree

        common_descendents = pd.DataFrame()
        altered_descendents = pd.DataFrame()

        # TODO: parallelize

        def __handle_common_child(common_child):
            _, common_child = common_child

            # return _compare_trees_rec(
            #    common_child.left_id, common_child.right_id, is_root=is_root
            # )

            # parallel

            _awaited_tasklist.append(
                (common_child.left_id, common_child.right_id, is_root)
            )

        if not common_children.empty:
            for row in common_children.iterrows():
                __handle_common_child(row)

        # recursive add altered subtrees
        for _, altered_child in altered_children.iterrows():
            if altered_child.rem_in_l:
                _a = _mark_sub_tree_altered(altered_child.id, right_tree, rem_in_l=True)
                altered_descendents = pd.concat([altered_descendents, _a], axis=0)

            elif altered_child.rem_in_r:
                _a = _mark_sub_tree_altered(altered_child.id, left_tree, rem_in_r=True)
                altered_descendents = pd.concat([altered_descendents, _a], axis=0)

            else:
                # _c, _a = _compare_trees_rec(
                #    altered_child.id, altered_child.other_id, is_root=False
                # )

                # parallel
                _awaited_tasklist.append(
                    (altered_child.id, altered_child.other_id, False)
                )

                # altered_descendents = pd.concat([altered_descendents, _a], axis=0)

                # if not _c.empty:
                #    _c.drop(
                #        columns=["processed", "score", "left_id", "right_id"],
                #        inplace=True,
                #    )

                #    _c["is_root"] = False

                #    altered_descendents = pd.concat([altered_descendents, _c], axis=0)[]

        awaited_tasks.put(
            (parent_pid, int(current_process().name), _awaited_tasklist), block=True
        )

        altered_children = pd.concat([altered_children, altered_descendents], axis=0)
        common_children = pd.concat([common_children, common_descendents], axis=0)

        if not altered_children.empty:
            altered_children = altered_children.drop(columns=["processed"])

        common_queue.put(common_children)
        edited_queue.put(altered_children)

        # awaited_tasks.cancel_join_thread()
        # edited_queue.cancel_join_thread()
        # common_queue.cancel_join_thread()

    MAX_NUM_PROCESSES = 10

    edited_queue = Queue()
    common_queue = Queue()
    awaited_tasks = Queue()
    # awaited_tasks.put((left_root, right_root, True))

    # while awaited_tasks.empty():
    #     pass

    alive_processes = []
    awaited_processes = []
    pending_tasks = {}
    new_process_id = 2

    altered_trees = []
    common_tree = [left_tree[left_tree["parent_id"] == -1]]

    # add the initial task
    pending_tasks = {0: {"awaiting": [1], "parent": None}}

    awaited_processes.append(
        Process(
            target=_compare_trees_rec,
            args=(
                left_root,
                right_root,
                True,
                edited_queue,
                common_queue,
                awaited_tasks,
                0,
            ),
            name="1",
        )
    )

    # exec loop
    while 0 in pending_tasks:
        # empty outputs queues

        if not edited_queue.empty():
            altered_trees.append(edited_queue.get())

        if not common_queue.empty():
            common_tree.append(common_queue.get())

        # handle awaited processes
        while len(alive_processes) <= MAX_NUM_PROCESSES and len(awaited_processes) != 0:
            p = awaited_processes.pop(0)
            alive_processes.append(p)
            p.start()

        # handle awaited tasks

        while not awaited_tasks.empty():
            parent_id, sender_id, task_list = awaited_tasks.get()

            awaited_processes.extend(
                [
                    Process(
                        target=_compare_trees_rec,
                        args=(
                            lr,
                            rr,
                            ir,
                            edited_queue,
                            common_queue,
                            awaited_tasks,
                            sender_id,
                        ),
                        name=f"{new_process_id + i}",
                    )
                    for i, (lr, rr, ir) in enumerate(task_list)
                ]
            )

            pending_tasks[sender_id] = {
                "awaiting": [new_process_id + i for i in range(len(task_list))],
                "parent": parent_id,
            }

            new_process_id += len(task_list)

        # handle dead processes
        for p in alive_processes:
            if not p.is_alive():
                p: Process

                sender_id = int(p.name)

                # if the process is not in the pending tasks then we ddnt read its output
                if sender_id not in pending_tasks:
                    continue

                # remove from parent if it has no pending subtasks
                if len(pending_tasks[sender_id]["awaiting"]) == 0:
                    parent_id = pending_tasks[sender_id]["parent"]
                    # if parent task has no more pending processes we need to cascade update all parent processes

                    while (
                        parent_id is not None
                        and len(pending_tasks[sender_id]["awaiting"]) == 0
                    ):
                        # remove from parent list

                        pending_tasks[parent_id]["awaiting"].remove(sender_id)

                        # remove from pending tasks
                        del pending_tasks[sender_id]

                        # check same for parent
                        sender_id = parent_id
                        parent_id = pending_tasks[sender_id]["parent"]

                    if parent_id == None:
                        del pending_tasks[sender_id]

                # remove from the alive processes
                alive_processes.remove(p)

    while not edited_queue.empty():
        altered_trees.append(edited_queue.get())

    while not common_queue.empty():
        common_tree.append(common_queue.get())

    altered_trees = pd.concat(altered_trees)

    altered_trees["visit_id"] = left_tree[left_tree["parent_id"] == -1].iloc[0].visit_id
    altered_trees["other_visit_id"] = (
        right_tree[right_tree["parent_id"] == -1].iloc[0].visit_id
    )

    # we still need to flip the altered rem_in_l
    flips = {
        "other_attr_class": "attr_class",
        "other_attr_id": "attr_id",
        "other_attr_src": "attr_src",
        "other_attributes": "attributes",
        "other_id": "id",
        "other_parent_id": "parent_id",
        "other_t_enter": "t_enter",
        "other_t_leave": "t_leave",
        "other_visual_cues": "visual_cues",
    }
    # print(altered_trees.columns)

    if not altered_trees.empty and not altered_trees[altered_trees["rem_in_l"] == True].empty:
        for col in flips:
            _val = altered_trees[altered_trees["rem_in_l"] == True][col]
            altered_trees.loc[
                altered_trees["rem_in_l"] == True, col
            ] = altered_trees.loc[altered_trees["rem_in_l"], flips[col]]
            altered_trees.loc[altered_trees["rem_in_l"] == True, flips[col]] = _val

    return pd.concat(common_tree), altered_trees.drop_duplicates()


def _get_dom_node_mappings(
    common_subtrees: pd.DataFrame,
    altered_subtrees: pd.DataFrame,
    browser_id_in,
    visit_id_in,
    browser_id_out,
    visit_id_out,
):
    map_dom_node_id_to_out = set()
    map_dom_node_id_to_in = set()
    is_edited = set()

    if not common_subtrees.empty:
        # common dom node mapping
        dom_node_id_in = common_subtrees.apply(
            lambda row: graph_node_id(browser_id_in, visit_id_in, row.left_id), axis=1
        ).values.tolist()
        dom_node_id_out = common_subtrees.apply(
            lambda row: graph_node_id(browser_id_out, visit_id_out, row.right_id),
            axis=1,
        ).values.tolist()

        map_dom_node_id_to_out = {x: y for x, y in zip(dom_node_id_in, dom_node_id_out)}
        map_dom_node_id_to_in = {y: x for x, y in zip(dom_node_id_in, dom_node_id_out)}

        is_edited = {x: False for x in dom_node_id_in + dom_node_id_out}

    if not altered_subtrees.empty:
        # edited dom nodes mapping
        dom_node_id_in = altered_subtrees.apply(
            lambda row: graph_node_id(browser_id_in, visit_id_in, row.id), axis=1
        ).values.tolist()
        dom_node_id_out = altered_subtrees.apply(
            lambda row: graph_node_id(browser_id_out, visit_id_out, row.other_id),
            axis=1,
        ).values.tolist()

        map_dom_node_id_to_out |= {
            x: y for x, y in zip(dom_node_id_in, dom_node_id_out)
        }
        map_dom_node_id_to_in |= {y: x for x, y in zip(dom_node_id_in, dom_node_id_out)}

        is_edited |= {x: True for x in dom_node_id_in + dom_node_id_out}

    return map_dom_node_id_to_out, map_dom_node_id_to_in, is_edited


def compare_graphs(
    common_subtrees: pd.DataFrame,
    altered_subtrees: pd.DataFrame,
    nodes_in: pd.DataFrame,
    edges_in: pd.DataFrame,
    nodes_out: pd.DataFrame,
    edges_out: pd.DataFrame,
    browser_id_in: str,
    visit_id_in: str,
    browser_id_out: str,
    visit_id_out: str,
):
    (
        map_dom_node_id_to_out,
        map_dom_node_id_to_in,
        is_dom_node_edited,
    ) = _get_dom_node_mappings(
        common_subtrees,
        altered_subtrees,
        browser_id_in,
        visit_id_in,
        browser_id_out,
        visit_id_out,
    )

    nodes_common = pd.DataFrame(
        columns=nodes_in.columns.tolist()
        + ["rem_in_r", "rem_in_l", "right_name", "left_name"]
    )
    edges_common = pd.DataFrame(columns=["src", "dst", "type", "value", "count"])

    nodes_removed = pd.DataFrame(
        columns=nodes_in.columns.tolist()
        + ["rem_in_r", "rem_in_l", "right_name", "left_name"]
    )
    edges = pd.DataFrame(columns=edges_in.columns.tolist() + ["rem_in_r", "rem_in_l"])

    node_ids_in = nodes_in["name"].values.tolist()
    node_ids_out = set(nodes_out["name"].values.tolist())

    node_ids_common = {}

    # print("! getting common nodes")

    for in_node in node_ids_in:
        if in_node in node_ids_out:
            node_ids_common[in_node] = in_node

        elif map_dom_node_id_to_out.get(in_node, None) in node_ids_out:
            node_ids_common[in_node] = map_dom_node_id_to_out[in_node]

    # print("! got common nodes. getting edited nodes.")

    nodes_common = nodes_in[nodes_in["name"].isin(node_ids_common)]

    if nodes_common.empty:
        nodes_common = pd.DataFrame(
            columns=nodes_common.columns.tolist()
            + ["right_name", "left_name", "rem_in_l", "rem_in_r"]
        )
    else:
        nodes_common["right_name"] = nodes_common.apply(
            lambda row: node_ids_common[row["name"]], axis=1
        )
        nodes_common["left_name"] = nodes_common["name"]
        nodes_common["rem_in_l"] = False
        nodes_common["rem_in_r"] = False

    # get the edited nodes

    nodes_rem_r = nodes_in[~nodes_in["name"].isin(node_ids_common)]
    nodes_rem_r["right_name"] = None
    nodes_rem_r["left_name"] = nodes_rem_r["name"].values
    nodes_rem_r["rem_in_r"] = True
    nodes_rem_r["rem_in_l"] = False

    nodes_rem_l = nodes_out[~nodes_out["name"].isin(node_ids_common)]
    nodes_rem_l["right_name"] = nodes_rem_l["name"].values
    nodes_rem_l["left_name"] = None
    nodes_rem_l["rem_in_l"] = True
    nodes_rem_l["rem_in_r"] = False

    nodes_removed = pd.concat([nodes_rem_l, nodes_rem_r])

    # print("!got nodes removed. getting edges")

    edges_to_add = []

    # for the nodes edited mark all edges as removed if they have source and destination edited

    edges_rem_r = [
        pd.DataFrame(
            columns=[
                "src",
                "dst",
                "type",
                "value",
                "count",
                "right_src",
                "right_dst",
                "right_value",
                "left_src",
                "left_dst",
                "left_value",
            ]
        )
    ]
    edges_rem_l = [
        pd.DataFrame(
            columns=[
                "src",
                "dst",
                "type",
                "value",
                "count",
                "right_src",
                "right_dst",
                "right_value",
                "left_src",
                "left_dst",
                "left_value",
            ]
        )
    ]

    # print("!getting edges removed from R. len(nodes_rem_r)=", len(nodes_rem_r))

    node_rem_r_names = set(nodes_rem_r["name"].unique().tolist())
    edges_rem_r.append(
        edges_in[
            ~(
                edges_in["src"].isin(node_rem_r_names)
                | edges_in["dst"].isin(node_rem_r_names)
            )
        ]
    )

    # print("!getting edges removed from L. len(nodes_rem_l)=", len(nodes_rem_l))

    node_rem_l_names = set(nodes_rem_l["name"].unique().tolist())
    edges_rem_l.append(
        edges_out[
            ~(
                edges_out["src"].isin(node_rem_l_names)
                | edges_out["dst"].isin(node_rem_l_names)
            )
        ]
    )

    edges_rem_r = pd.concat(edges_rem_r)
    edges_rem_l = pd.concat(edges_rem_l)

    edges_rem_r["rem_in_r"] = True
    edges_rem_r["rem_in_l"] = False
    edges_rem_r["right_src"] = None
    edges_rem_r["right_dst"] = None
    edges_rem_r["right_value"] = None
    edges_rem_r["left_src"] = edges_rem_r["src"].values
    edges_rem_r["left_dst"] = edges_rem_r["dst"].values
    edges_rem_r["left_value"] = edges_rem_r["value"].values
    edges_rem_r["dcount"] = 0
    edges_rem_r.rename(columns={"dcount": "count", "count": "dcount"}, inplace=True)

    edges_rem_l["rem_in_r"] = False
    edges_rem_l["rem_in_l"] = True
    edges_rem_l["left_src"] = None
    edges_rem_l["left_dst"] = None
    edges_rem_l["left_value"] = None
    edges_rem_l["right_src"] = edges_rem_l["src"].values
    edges_rem_l["right_dst"] = edges_rem_l["dst"].values
    edges_rem_l["right_value"] = edges_rem_l["value"].values
    edges_rem_l["dcount"] = 0
    edges_rem_l.rename(columns={"dcount": "count", "count": "dcount"}, inplace=True)

    edges_to_add.append(edges_rem_r)
    edges_to_add.append(edges_rem_l)

    # print("!got edges removed and adding. getting common edges.")

    edge_ids_in = [tuple(x) for x in edges_in[["src", "dst", "value"]].values.tolist()]

    edge_ids_out = set(
        [tuple(x) for x in edges_out[["src", "dst", "value"]].values.tolist()]
    )

    # edge_ids_common = edge_ids_in & edge_ids_out

    edge_ids_common = {}

    for in_edge in edge_ids_in:
        mapped_dst = (in_edge[0], map_dom_node_id_to_out.get(in_edge[1]), in_edge[2])
        mapped_src = (map_dom_node_id_to_out.get(in_edge[0]), in_edge[1], in_edge[2])
        mapped_both = (
            map_dom_node_id_to_out.get(in_edge[0]),
            map_dom_node_id_to_out.get(in_edge[1]),
            in_edge[2],
        )

        if in_edge in edge_ids_out:
            edge_ids_common[in_edge] = in_edge

        elif mapped_dst in edge_ids_out:
            edge_ids_common[in_edge] = mapped_dst

        elif mapped_src in edge_ids_out:
            edge_ids_common[in_edge] = mapped_src

        elif mapped_both in edge_ids_out:
            edge_ids_common[in_edge] = mapped_both

    # print("!got common edged")

    edges_common = edges_in[
        edges_in.apply(lambda x: tuple([x.src, x.dst, x.value]), axis=1).isin(
            edge_ids_common
        )
    ]

    edges_only_in = edges_in[
        ~edges_in.apply(lambda x: tuple([x.src, x.dst, x.value]), axis=1).isin(
            edge_ids_common
        )
    ]

    edges_only_out = edges_out[
        ~edges_out.apply(lambda x: tuple([x.src, x.dst, x.value]), axis=1).isin(
            edge_ids_common.values()
        )
    ]

    edges_only_in["rem_in_r"] = True
    edges_only_in["rem_in_l"] = False
    edges_only_in["right_src"] = None
    edges_only_in["right_dst"] = None
    edges_only_in["right_value"] = None
    edges_only_in["left_src"] = edges_only_in["src"].values
    edges_only_in["left_dst"] = edges_only_in["dst"].values
    edges_only_in["left_value"] = edges_only_in["value"].values
    edges_only_in["dcount"] = 0
    edges_only_in.rename(columns={"dcount": "count", "count": "dcount"}, inplace=True)

    edges_to_add.append(edges_only_in)

    edges_only_out["rem_in_r"] = False
    edges_only_out["rem_in_l"] = True
    edges_only_out["left_src"] = None
    edges_only_out["left_dst"] = None
    edges_only_out["left_value"] = None
    edges_only_out["right_src"] = edges_only_out["src"].values
    edges_only_out["right_dst"] = edges_only_out["dst"].values
    edges_only_out["right_value"] = edges_only_out["value"].values
    edges_only_out["dcount"] = 0
    edges_only_out.rename(columns={"dcount": "count", "count": "dcount"}, inplace=True)

    edges_to_add.append(edges_only_out)

    if edges_common.empty:
        edges_common = pd.DataFrame(
            columns=[
                "src",
                "dst",
                "type",
                "value",
                "count",
                "dcount",
                "rem_in_l",
                "rem_in_r",
                "right_src",
                "right_dst",
                "right_value",
                "left_src",
                "left_dst",
                "left_value",
            ]
        )

        edges_count_changed = pd.DataFrame(
            columns=[
                "src",
                "dst",
                "type",
                "value",
                "count",
                "dcount",
                "rem_in_l",
                "rem_in_r",
                "right_src",
                "right_dst",
                "right_value",
                "left_src",
                "left_dst",
                "left_value",
            ]
        )

    else:
        edges_common[["right_src", "right_dst", "right_value"]] = edges_common.apply(
            lambda row: edge_ids_common[(row.src, row.dst, row.value)],
            axis=1,
            result_type="expand",
        )

        edges_common[["left_src", "left_dst", "left_value"]] = edges_common[
            ["src", "dst", "value"]
        ]

        edges_commonout = edges_out[
            edges_out.apply(lambda x: tuple([x.src, x.dst, x.value]), axis=1).isin(
                edge_ids_common.values()
            )
        ]

        edges_common = edges_common.merge(
            edges_commonout,
            left_on=["right_src", "right_dst", "right_value"],
            right_on=["src", "dst", "value"],
        )

        if "count" not in edges_common.columns:
            edges_common = pd.DataFrame(
                columns=edges_common.columns.tolist() + ["count_x"]
            )

        if "count" in edges_common.columns:
            edges_common.rename(columns={"count": "count_x"}, inplace=True)

        if "count_y" not in edges_common.columns:
            edges_common = pd.DataFrame(
                columns=edges_common.columns.tolist() + ["count_y"]
            )

        # for the common

        edges_count_changed = edges_common[
            edges_common.apply(lambda row: row.count_x != row.count_y, axis=1)
        ]
        edges_common = edges_common[
            edges_common.apply(lambda row: row.count_x == row.count_y, axis=1)
        ]

        if not edges_count_changed.empty:
            edges_count_changed["dcount"] = edges_count_changed.apply(
                lambda row: row.count_y - row.count_x, axis=1
            )

            edges_count_changed.drop(columns=["count_y"], inplace=True)
            edges_count_changed.rename(columns={"count_x": "count"}, inplace=True)
            edges_count_changed["rem_in_l"] = False
            edges_count_changed["rem_in_r"] = False

            edges_to_add.append(edges_count_changed)

        if not edges_common.empty:
            edges_common.drop(columns=["count_y"], inplace=True)
            edges_common.rename(columns={"count_x": "count"}, inplace=True)

            edges_common["rem_in_l"] = False
            edges_common["rem_in_r"] = False
            edges_common["dcount"] = 0

            edges_to_add.append(edges_common)

    edges = pd.concat(edges_to_add)
    edges = edges[edges["count"] != 0 | edges["rem_in_l"] | edges["rem_in_r"]]

    edges = edges.drop_duplicates()

    nodes = pd.concat([nodes_common, nodes_removed])
    nodes = nodes.drop_duplicates()

    return nodes, edges
