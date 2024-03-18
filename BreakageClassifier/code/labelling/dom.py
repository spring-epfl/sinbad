import pandas as pd

from BreakageClassifier.code.features.utils import time_it

def _duplicates_in_features(df: pd.DataFrame):
    # "nodeName": "tag",
    #     "rem_in_l": "added",
    #     "rem_in_r": "removed",
    #     "browser_id": "prev_bid",
    #     "visit_id": "prev_vid",
    #     "id": "prev_id",
    #     "attributes": "prev_attr",
    #     "visual_cues": "prev_vc",
    #     "parent_id": "prev_parent",
    #     "attr_id": "prev_attr_id",
    #     "attr_class": "prev_attr_class",
    #     "attr_src": "prev_attr_src",
    #     "t_enter": "prev_te",
    #     "t_leave": "prev_tl",
    #     "other_attr_class": "new_attr_class",
    #     "other_attr_id": "new_attr_id",
    #     "other_attr_src": "new_attr_src",
    #     "other_attributes": "new_attr",
    #     "other_browser_id": "new_bid",
    #     "other_visit_id": "new_vid",
    #     "other_id": "new_id",
    #     "other_parent_id": "new_parent",
    #     "other_visual_cues": "new_vc",
    #     "other_t_enter": "new_te",
    #     "other_t_leave": "new_tl",
    df_features = df.drop(
        colummns = [
            'rem_in_l',
            'rem_in_r',
            'browser_id',
            'visit_id',
            'id',
            'parent_id',
            ''
        ]
    )

def _is_edited_in_both(df_1: pd.DataFrame, df_2: pd.DataFrame, id_col):
    ids_1 = set(df_1[id_col])
    ids_2 = set(df_2[id_col])

    ids_common = ids_1.intersection(ids_2)

    return df_1[id_col].isin(ids_common)


def _add_children(root_df: pd.DataFrame, df: pd.DataFrame):
    _all_children = [
        root_df,
    ]

    for _, root in root_df.iterrows():
        _children = df[
            (df["other_visit_id"] == root.other_visit_id)
            & (df["visit_id"] == root.visit_id)
            & (
                (
                    (df["other_t_enter"] > root.other_t_enter)
                    & (df["other_t_leave"] < root.other_t_leave)
                )
                | ((df["t_enter"] > root.t_enter) & (df["t_leave"] < root.t_leave))
            )
        ].copy()

        _all_children.append(_children)

    return pd.concat(_all_children, axis=0)


@time_it
def label_alterations(
    issue: pd.Series,
    no_to_fixed_common: pd.DataFrame,
    no_to_fixed_diff: pd.DataFrame,
    fixed_to_broken_diff: pd.DataFrame,
    no_to_broken_diff: pd.DataFrame,
):
    col_name_map_prev_new = {
        "nodeName": "tag",
        "rem_in_l": "added",
        "rem_in_r": "removed",
        "browser_id": "prev_bid",
        "visit_id": "prev_vid",
        "id": "prev_id",
        "attributes": "prev_attr",
        "visual_cues": "prev_vc",
        "parent_id": "prev_parent",
        "attr_id": "prev_attr_id",
        "attr_class": "prev_attr_class",
        "attr_src": "prev_attr_src",
        "t_enter": "prev_te",
        "t_leave": "prev_tl",
        "other_attr_class": "new_attr_class",
        "other_attr_id": "new_attr_id",
        "other_attr_src": "new_attr_src",
        "other_attributes": "new_attr",
        "other_browser_id": "new_bid",
        "other_visit_id": "new_vid",
        "other_id": "new_id",
        "other_parent_id": "new_parent",
        "other_visual_cues": "new_vc",
        "other_t_enter": "new_te",
        "other_t_leave": "new_tl",
    }

    col_name_map_new_prev = {
        "nodeName": "tag",
        "rem_in_l": "removed",
        "rem_in_r": "added",
        "browser_id": "new_bid",
        "visit_id": "new_vid",
        "id": "new_id",
        "attributes": "new_attr",
        "visual_cues": "new_vc",
        "parent_id": "new_parent",
        "attr_id": "new_attr_id",
        "attr_class": "new_attr_class",
        "attr_src": "new_attr_src",
        "t_enter": "new_te",
        "t_leave": "new_tl",
        "other_attr_class": "prev_attr_class",
        "other_attr_id": "prev_attr_id",
        "other_attr_src": "prev_attr_src",
        "other_attributes": "prev_attr",
        "other_browser_id": "prev_bid",
        "other_visit_id": "prev_vid",
        "other_id": "prev_id",
        "other_parent_id": "prev_parent",
        "other_visual_cues": "prev_vc",
        "other_t_enter": "prev_te",
        "other_t_leave": "prev_tl",
    }

    edits_df_list = []

    ## RULE get all no->fixed from fixed (rm_in_r)  => label GOOD

    if not no_to_fixed_diff.empty:
        no_fixed_rem_edit: pd.DataFrame = no_to_fixed_diff[
            (no_to_fixed_diff["is_root"] == True)
            & (
                (no_to_fixed_diff["rem_in_r"] == True)
            )
        ]

        no_fixed_rem_edit = _add_children(no_fixed_rem_edit, no_to_fixed_diff)

        no_fixed_rem_edit.rename(columns=col_name_map_prev_new, inplace=True)

        no_fixed_rem_edit["is_breaking"] = -1
        no_fixed_rem_edit["origin"] = "no_fixed"
        no_fixed_rem_edit["flipped"] = False

        edits_df_list.append(no_fixed_rem_edit)

    # get all no->fixed added (rm_in_l) => label NEUTRAL
    
    # those edited can either be GOOD or NEUTRAL so ignore

    if not no_to_fixed_diff.empty:
        no_fixed_add: pd.DataFrame = no_to_fixed_diff[
            (no_to_fixed_diff["is_root"] == True)
            & ((no_to_fixed_diff["rem_in_l"] == True))
        ]

        no_fixed_add = _add_children(no_fixed_add, no_to_fixed_diff)

        no_fixed_add.rename(columns=col_name_map_prev_new, inplace=True)

        no_fixed_add["origin"] = "no_fixed"
        no_fixed_add["flipped"] = False
        no_fixed_add["is_breaking"] = 0

        edits_df_list.append(no_fixed_add)

    # get all fixed->broken removed (rm_in_r) and not removed in fixed->no    => label BROKEN

    if not fixed_to_broken_diff.empty and not no_to_fixed_diff.empty:
        fixed_broken_rem: pd.DataFrame = fixed_to_broken_diff[
            (fixed_to_broken_diff["is_root"] == True)
            & ((fixed_to_broken_diff["rem_in_r"] == True))
        ]

        # remove if removed in fixed->no i.e added in no->fixed
        fbr_index = fixed_broken_rem.set_index(["browser_id", "visit_id", "id"]).index

        nfa_index = no_fixed_add.set_index(["new_bid", "new_vid", "new_id"]).index

        fixed_broken_rem = fixed_broken_rem.loc[~fbr_index.isin(nfa_index)]

        fixed_broken_rem = _add_children(fixed_broken_rem, fixed_to_broken_diff)

        
        fixed_broken_rem.rename(columns=col_name_map_prev_new, inplace=True)
        fixed_broken_rem["origin"] = "fixed_broken"
        fixed_broken_rem["flipped"] = False
        fixed_broken_rem["is_breaking"] = 1

        edits_df_list.append(fixed_broken_rem)

    # get all fixed->broken added (rm_in_l)         => flip operations => label GOOD

    if not fixed_to_broken_diff.empty:
        fixed_broken_add: pd.DataFrame = fixed_to_broken_diff[
            (fixed_to_broken_diff["is_root"] == True)
            & ((fixed_to_broken_diff["rem_in_l"] == True))
        ]

        fixed_broken_add = _add_children(fixed_broken_add, fixed_to_broken_diff)

        fixed_broken_add.rename(columns=col_name_map_new_prev, inplace=True)

        # fixed_broken_add.rename(
        #     columns={"added": "removed", "removed": "added"},
        #     inplace=True,
        # )

        fixed_broken_add["origin"] = "fixed_broken"
        fixed_broken_add["flipped"] = True
        fixed_broken_add["is_breaking"] = -1

        edits_df_list.append(fixed_broken_add)

    # get all fixed->broken edited AND no change in no->fixed => label BREAK

    if not fixed_to_broken_diff.empty:
        fixed_broken_edit: pd.DataFrame = fixed_to_broken_diff[
            (fixed_to_broken_diff["is_root"] == True)
            & ((fixed_to_broken_diff["edited"] == True))
        ]

        no_fixed_common = set()

        if not no_to_fixed_common.empty:
            no_fixed_common = set(no_to_fixed_common["right_id"])

        fixed_broken_edit_no_change = fixed_broken_edit[
            fixed_broken_edit["id"].isin(no_fixed_common)
        ]

        fixed_broken_edit_no_change = _add_children(
            fixed_broken_edit_no_change, fixed_to_broken_diff
        )

        fixed_broken_edit_no_change.rename(
            columns=col_name_map_prev_new,
            inplace=True,
        )

        fixed_broken_edit_no_change["origin"] = "fixed_broken"
        fixed_broken_edit_no_change["flipped"] = False
        fixed_broken_edit_no_change["is_breaking"] = 1

        edits_df_list.append(fixed_broken_edit_no_change)

    # get all no->broken edited and EDIT differently or not edited in no->fixed => label BROKEN

    if not no_to_broken_diff.empty and not fixed_to_broken_diff.empty:
        no_broken_edit: pd.DataFrame = no_to_broken_diff[
            (no_to_broken_diff["is_root"] == True)
            & ((no_to_broken_diff["edited"] == True))
        ]

        no_broken_edit_no_change = no_broken_edit[
            (no_broken_edit["id"].isin(no_fixed_common))
            | _is_edited_in_both(no_broken_edit, fixed_broken_edit, "other_id")
        ].copy(deep=True)

        no_broken_edit_no_change = _add_children(
            no_broken_edit_no_change,
            no_to_broken_diff,
        )

        no_broken_edit_no_change.rename(
            columns=col_name_map_prev_new,
            inplace=True,
        )

        no_broken_edit_no_change["origin"] = "no_broken"
        no_broken_edit_no_change["flipped"] = False
        no_broken_edit_no_change["is_breaking"] = 1

        edits_df_list.append(no_broken_edit_no_change)

    # get all no->broken edited and EDITED SAME WAY in no->fixed => label GOOD

    if (
        not no_to_fixed_diff.empty
        and not no_to_broken_diff.empty
        and not fixed_to_broken_diff.empty
    ):
        no_fixed_edit: pd.DataFrame = no_to_fixed_diff[
            (no_to_fixed_diff["is_root"] == True)
            & ((no_to_fixed_diff["edited"] == True))
        ]

        # TODO: edit this so that we check if the whole subgraph is compliant

        no_broken_edit_same_change = no_broken_edit.merge(
            no_fixed_edit,
            how="inner",
            on=["visit_id", "id", "other_visit_id", "other_id"],
            suffixes=("", "_y"),
        )

        no_broken_edit_same_change = no_broken_edit_same_change[no_broken_edit.columns]

        no_broken_edit_same_change = _add_children(
            no_broken_edit_same_change, no_to_fixed_diff
        )

        no_broken_edit_same_change.rename(
            columns=col_name_map_prev_new,
            inplace=True,
        )

        no_broken_edit_same_change["is_breaking"] = -1
        no_broken_edit_same_change["origin"] = "no_broken"
        no_broken_edit_same_change["flipped"] = False

        edits_df_list.append(no_broken_edit_same_change)

    if len(edits_df_list) == 0:
        return pd.DataFrame()

    alterations = pd.concat(edits_df_list)

    alterations["is_root"] = alterations["is_root"] == True
    alterations["added"] = alterations["added"] == True
    alterations["edited"] = alterations["edited"] == True
    alterations["removed"] = alterations["removed"] == True

    alterations["issue"] = issue.issue_id

    return alterations



@time_it
def label_alterations_neg(
    issue: pd.Series,
    no_to_after_common: pd.DataFrame,
    no_to_after_diff: pd.DataFrame,
    after_to_before_diff: pd.DataFrame,
    no_to_before_diff: pd.DataFrame,
):
    col_name_map_prev_new = {
        "nodeName": "tag",
        "rem_in_l": "added",
        "rem_in_r": "removed",
        "browser_id": "prev_bid",
        "visit_id": "prev_vid",
        "id": "prev_id",
        "attributes": "prev_attr",
        "visual_cues": "prev_vc",
        "parent_id": "prev_parent",
        "attr_id": "prev_attr_id",
        "attr_class": "prev_attr_class",
        "attr_src": "prev_attr_src",
        "t_enter": "prev_te",
        "t_leave": "prev_tl",
        "other_attr_class": "new_attr_class",
        "other_attr_id": "new_attr_id",
        "other_attr_src": "new_attr_src",
        "other_attributes": "new_attr",
        "other_browser_id": "new_bid",
        "other_visit_id": "new_vid",
        "other_id": "new_id",
        "other_parent_id": "new_parent",
        "other_visual_cues": "new_vc",
        "other_t_enter": "new_te",
        "other_t_leave": "new_tl",
    }

    col_name_map_new_prev = {
        "nodeName": "tag",
        "rem_in_l": "removed",
        "rem_in_r": "added",
        "browser_id": "new_bid",
        "visit_id": "new_vid",
        "id": "new_id",
        "attributes": "new_attr",
        "visual_cues": "new_vc",
        "parent_id": "new_parent",
        "attr_id": "new_attr_id",
        "attr_class": "new_attr_class",
        "attr_src": "new_attr_src",
        "t_enter": "new_te",
        "t_leave": "new_tl",
        "other_attr_class": "prev_attr_class",
        "other_attr_id": "prev_attr_id",
        "other_attr_src": "prev_attr_src",
        "other_attributes": "prev_attr",
        "other_browser_id": "prev_bid",
        "other_visit_id": "prev_vid",
        "other_id": "prev_id",
        "other_parent_id": "prev_parent",
        "other_visual_cues": "prev_vc",
        "other_t_enter": "prev_te",
        "other_t_leave": "prev_tl",
    }

    edits_df_list = []

    ## RULE get all no->after removed from after (rm_in_r)  => label GOOD

    if not no_to_after_diff.empty:
        no_after_rem: pd.DataFrame = no_to_after_diff[
            (no_to_after_diff["is_root"] == True)
            & (
                (no_to_after_diff["rem_in_r"] == True)
            )
        ]

        no_after_rem = _add_children(no_after_rem, no_to_after_diff)

        no_after_rem.rename(columns=col_name_map_prev_new, inplace=True)

        no_after_rem["is_breaking"] = -1
        no_after_rem["origin"] = "no_after"
        no_after_rem["flipped"] = False

        edits_df_list.append(no_after_rem)

    # get all no -> after added (rm_in_l) => label NEUTRAL
    
    # those edited can either be GOOD or NEUTRAL so ignore

    if not no_to_after_diff.empty:
        no_after_add: pd.DataFrame = no_to_after_diff[
            (no_to_after_diff["is_root"] == True)
            & ((no_to_after_diff["rem_in_l"] == True))
        ]

        no_after_add = _add_children(no_after_add, no_to_after_diff)

        _no_after_add_not_renamed = no_after_add.copy(deep=True)
        no_after_add.rename(columns=col_name_map_prev_new, inplace=True)

        no_after_add["origin"] = "no_after"
        no_after_add["flipped"] = False
        no_after_add["is_breaking"] = 0

        edits_df_list.append(no_after_add)

    # get all after->before added (rm_in_l) => flip operations => label GOOD

    if not after_to_before_diff.empty:
        after_before_add: pd.DataFrame = after_to_before_diff[
            (after_to_before_diff["is_root"] == True)
            & ((after_to_before_diff["rem_in_l"] == True))
        ]

        after_before_add = _add_children(after_before_add, after_to_before_diff)

        after_before_add.rename(columns=col_name_map_new_prev, inplace=True)

        after_before_add["origin"] = "after_before"
        after_before_add["flipped"] = True
        after_before_add["is_breaking"] = -1

        edits_df_list.append(after_before_add)

    # get all after->before edited AND no change in no->after => flip operations => label GOOD

    if not after_to_before_diff.empty:
        fixed_broken_edit: pd.DataFrame = after_to_before_diff[
            (after_to_before_diff["is_root"] == True)
            & ((after_to_before_diff["edited"] == True))
        ]

        no_fixed_common = set()

        if not no_to_after_common.empty:
            no_fixed_common = set(no_to_after_common["right_id"])

        after_before_edit_no_change = fixed_broken_edit[
            fixed_broken_edit["id"].isin(no_fixed_common)
        ]

        after_before_edit_no_change = _add_children(
            after_before_edit_no_change, after_to_before_diff
        )

        after_before_edit_no_change.rename(
            columns=col_name_map_new_prev,
            inplace=True,
        )

        after_before_edit_no_change["origin"] = "after_before"
        after_before_edit_no_change["flipped"] = True
        after_before_edit_no_change["is_breaking"] = -1

        edits_df_list.append(after_before_edit_no_change)

    # get all no->before edited and EDITED SAME WAY in no->after => label GOOD

    if (
        not no_to_after_diff.empty
        and not no_to_before_diff.empty
        and not after_to_before_diff.empty
    ):
        no_after_edit: pd.DataFrame = no_to_after_diff[
            (no_to_after_diff["is_root"] == True)
            & ((no_to_after_diff["edited"] == True))
        ]

        # TODO: edit this so that we check if the whole subgraph is compliant

        if not no_after_edit.empty:

            no_before_edit_same_change = _no_after_add_not_renamed.merge(
                no_after_edit,
                how="inner",
                on=["visit_id", "id", "other_visit_id", "other_id"],
                suffixes=("", "_y"),
            )

            no_before_edit_same_change = no_before_edit_same_change[no_after_edit.columns]

            no_before_edit_same_change = _add_children(
                no_before_edit_same_change, no_to_after_diff
            )

            no_before_edit_same_change.rename(
                columns=col_name_map_prev_new,
                inplace=True,
            )

            no_before_edit_same_change["is_breaking"] = -1
            no_before_edit_same_change["origin"] = "no_before"
            no_before_edit_same_change["flipped"] = False

            edits_df_list.append(no_before_edit_same_change)

    if len(edits_df_list) == 0:
        return pd.DataFrame()

    alterations = pd.concat(edits_df_list)

    alterations["is_root"] = alterations["is_root"] == True
    alterations["added"] = alterations["added"] == True
    alterations["edited"] = alterations["edited"] == True
    alterations["removed"] = alterations["removed"] == True

    alterations["issue"] = issue.issue_id

    return alterations



def select_subtrees(
    df: pd.DataFrame,
    root_df: pd.DataFrame,
    condition: pd.DataFrame,
    origin: str,
    flipped: bool,
):
    col_name_map_prev_new = {
        "nodeName": "tag",
        "rem_in_l": "added",
        "rem_in_r": "removed",
        "browser_id": "prev_bid",
        "visit_id": "prev_vid",
        "id": "prev_id",
        "attributes": "prev_attr",
        "visual_cues": "prev_vc",
        "parent_id": "prev_parent",
        "attr_id": "prev_attr_id",
        "attr_class": "prev_attr_class",
        "attr_src": "prev_attr_src",
        "t_enter": "prev_te",
        "t_leave": "prev_tl",
        "other_attr_class": "new_attr_class",
        "other_attr_id": "new_attr_id",
        "other_attr_src": "new_attr_src",
        "other_attributes": "new_attr",
        "other_browser_id": "new_bid",
        "other_visit_id": "new_vid",
        "other_id": "new_id",
        "other_parent_id": "new_parent",
        "other_visual_cues": "new_vc",
        "other_t_enter": "new_te",
        "other_t_leave": "new_tl",
    }

    col_name_map_new_prev = {
        "nodeName": "tag",
        "rem_in_l": "removed",
        "rem_in_r": "added",
        "browser_id": "new_bid",
        "visit_id": "new_vid",
        "id": "new_id",
        "attributes": "new_attr",
        "visual_cues": "new_vc",
        "parent_id": "new_parent",
        "attr_id": "new_attr_id",
        "attr_class": "new_attr_class",
        "attr_src": "new_attr_src",
        "t_enter": "new_te",
        "t_leave": "new_tl",
        "other_attr_class": "prev_attr_class",
        "other_attr_id": "prev_attr_id",
        "other_attr_src": "prev_attr_src",
        "other_attributes": "prev_attr",
        "other_browser_id": "prev_bid",
        "other_visit_id": "prev_vid",
        "other_id": "prev_id",
        "other_parent_id": "prev_parent",
        "other_visual_cues": "prev_vc",
        "other_t_enter": "prev_te",
        "other_t_leave": "prev_tl",
    }

    df_out = root_df[condition].copy()
    df_out = _add_children(df_out, df)
    df_out["origin"] = origin
    df_out["flipped"] = flipped

    if flipped:
        df_out.rename(columns=col_name_map_new_prev, inplace=True)
    else:
        df_out.rename(columns=col_name_map_prev_new, inplace=True)

    return df_out


def get_difference(df, df_remove):
    merged_df = pd.merge(df, df_remove, indicator=True, how="outer")
    result_df = merged_df[merged_df["_merge"] == "left_only"]
    result_df.drop(columns="_merge", inplace=True)
    return result_df


def label_alterations2(
    no_to_fixed_common: pd.DataFrame,
    no_to_fixed_diff: pd.DataFrame,
    fixed_to_broken_diff: pd.DataFrame,
    no_to_broken_diff: pd.DataFrame,
):
    REVERSE_COLS = {
        "removed": "added",
        "added": "removed",
        "new_bid": "prev_bid",
        "new_vid": "prev_vid",
        "new_id": "prev_id",
        "new_attr": "prev_attr",
        "new_vc": "prev_vc",
        "new_parent": "prev_parent",
        "new_attr_id": "prev_attr_id",
        "new_attr_class": "prev_attr_class",
        "new_attr_src": "prev_attr_src",
        "new_te": "prev_te",
        "new_tl": "prev_tl",
        "prev_attr_class": "new_attr_class",
        "prev_attr_id": "new_attr_id",
        "prev_attr_src": "new_attr_src",
        "prev_attr": "new_attr",
        "prev_bid": "new_bid",
        "prev_vid": "new_vid",
        "prev_id": "new_id",
        "prev_parent": "new_parent",
        "prev_vc": "new_vc",
        "prev_te": "new_te",
        "prev_tl": "new_tl",
    }

    EDIT_NODE_ID_COLS = [
        "prev_bid",
        "prev_vid",
        "prev_id",
        "new_bid",
        "new_vid",
        "new_id",
    ]

    # Get all the roots
    NF_roots = no_to_fixed_diff[no_to_fixed_diff["is_root"] == True]
    NB_roots = no_to_broken_diff[no_to_broken_diff["is_root"] == True]
    FB_roots = fixed_to_broken_diff[fixed_to_broken_diff["is_root"] == True]

    edits_to_out = []

    ### Fixing edits

    # 1. removed or edited in NF
    if not NF_roots.empty:
        NF_rem = select_subtrees(
            no_to_fixed_diff, NF_roots, NF_roots["rem_in_r"] == True, "no->fixed", False
        )

        edits_to_out.append(NF_rem)

        NF_edit = select_subtrees(
            no_to_fixed_diff, NF_roots, NF_roots["edited"] == True, "no->fixed", False
        )

        edits_to_out.append(NF_edit)

    # .. removed or edited in BF
    if not FB_roots.empty:
        BF_rem = select_subtrees(
            fixed_to_broken_diff,
            FB_roots,
            FB_roots["rem_in_l"] == True,
            "fixed->broken",
            True,
        )

        edits_to_out.append(BF_rem)

        BF_edit = select_subtrees(
            fixed_to_broken_diff,
            FB_roots,
            FB_roots["edited"] == True,
            "fixed->broken",
            True,
        )

        edits_to_out.append(BF_edit)

    # 2. removed or edited in NB and NF
    if not NB_roots.empty:
        NB_rem = select_subtrees(
            no_to_broken_diff,
            NB_roots,
            NB_roots["rem_in_r"] == True,
            "no->broken",
            False,
        )

        NB_edit = select_subtrees(
            no_to_broken_diff,
            NB_roots,
            NB_roots["edited"] == True,
            "no->broken",
            False,
        )

    # merge with those in NF
    if not NF_rem.empty and not NB_rem.empty:
        NF_NB_rem = NF_rem[EDIT_NODE_ID_COLS].merge(
            NB_rem, how="inner", on=EDIT_NODE_ID_COLS
        )

        edits_to_out.append(NF_NB_rem)

    if not NF_edit.empty and not NB_edit.empty:
        NF_NB_edit = NF_edit[EDIT_NODE_ID_COLS].merge(
            NB_edit, how="inner", on=EDIT_NODE_ID_COLS
        )

        edits_to_out.append(NF_NB_edit)

    # label them
    if len(edits_to_out):
        fixing_edits = pd.concat(edits_to_out)
        fixing_edits["is_breaking"] = -1
    else:
        fixing_edits = pd.DataFrame()

    ### Breaking edits

    edits_to_out = []

    # 1. removed in NB but not in NF
    if (
        not NB_roots.empty
        and not NB_rem.empty
        and not NF_roots.empty
        and not NF_rem.empty
    ):
        # print(len(NB_rem), len(NF_rem))
        NB_not_NF_rem = get_difference(NB_rem, NF_rem[EDIT_NODE_ID_COLS])
        edits_to_out.append(NB_not_NF_rem)

    elif not NB_rem.empty and NF_rem.empty:
        NB_not_NF_rem = NB_rem
        edits_to_out.append(NB_not_NF_rem)

    # 2. edited // //

    if not NB_rem.empty and not NF_rem.empty:
        NB_not_NF_edit = get_difference(NB_edit, NF_edit[EDIT_NODE_ID_COLS])
        edits_to_out.append(NB_not_NF_edit)

    elif not NB_rem.empty and NF_rem.empty:
        NB_not_NF_rem = NB_rem
        edits_to_out.append(NB_not_NF_rem)

    # 3. added in BF but not in NF

    if not FB_roots.empty:
        BF_add = select_subtrees(
            fixed_to_broken_diff,
            FB_roots,
            FB_roots["rem_in_r"] == True,
            "fixed->broken",
            True,
        )

    if not NF_roots.empty:
        NF_add = select_subtrees(
            no_to_fixed_diff,
            NF_roots,
            NF_roots["rem_in_l"] == True,
            "no->fixed",
            False,
        )

    if not FB_roots.empty and not NF_roots.empty:
        BF_not_NF_add = get_difference(BF_add, NF_add[EDIT_NODE_ID_COLS])
        reverse_BF_not_NF_add = BF_not_NF_add.rename(columns=REVERSE_COLS, inplace=True)

        edits_to_out.append(reverse_BF_not_NF_add)

    if len(edits_to_out):
        breaking_edits = pd.concat(edits_to_out)
        breaking_edits["is_breaking"] = 1
    else:
        breaking_edits = pd.DataFrame()

    ### Neutral edits

    # 1. added in NF
    if not NF_roots.empty:
        edits_to_out.append(NF_add)

    # .. added in NB
    if not NB_roots.empty:
        NB_add = select_subtrees(
            no_to_broken_diff,
            NB_roots,
            NB_roots["rem_in_l"] == True,
            "no->broken",
            False,
        )

        edits_to_out.append(NB_add)

    # 2. added in BF and NF
    if (
        not FB_roots.empty
        and not BF_add.empty
        and not NF_roots.empty
        and not NF_add.empty
    ):
        BF_add = BF_add.merge(
            NF_add[EDIT_NODE_ID_COLS], how="inner", on=EDIT_NODE_ID_COLS
        )

        edits_to_out.append(BF_add)

    if len(edits_to_out):
        neutral_edits = pd.concat(edits_to_out)
        neutral_edits["is_breaking"] = 0
    else:
        neutral_edits = pd.DataFrame()

    return pd.concat([breaking_edits, neutral_edits, fixing_edits])
