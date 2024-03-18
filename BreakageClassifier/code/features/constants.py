SUBTREE_STRUCTURAL_FEATURES = [
    # subtree structural
    "edited",
    "added",
    "removed",
    "n_nodes",
    "tree_height",
    "mean_degree",
    "std_degree",
]


SUBTREE_VISUAL_FEATURES = [
    # subtree visual
    # "saliency",
    "n_salient",
    "n_salient_add",
    "n_salient_rem",
    "n_salient_ed",
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


SUBTREE_CONTENT_FEATURES = [
    # subtree content
    "is_third_party",
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
    "num_url_queries",
    "num_id_in_query_field",
    "num_url_params",
    "num_id_in_param_field",
    "base_domain_in_query",
    "semicolon_in_query",
    "screen_size_present",
    "ad_size_present",
    "ad_size_in_qs_present",
    "keyword_raw_present",
    "keyword_char_present",
]


SUBTREE_FUNCTIONAL_FEATURES = [
    # subtree functional
    "num_requests",
    "n_el_in_scr_tree",
    "dn_el_in_scr_tree",
    "dn_err_in_scr_tree",
    "n_scr_in_el_tree",
    "dn_scr_in_el_tree",
    "dn_err_in_int_tree",
    "dn_el_in_int_tree",
    "n_int_in_tree",
    "n_req_in_scr_tree",
    "dn_req_in_scr_tree",
]

SUBTREE_FEATURES = [
    *SUBTREE_STRUCTURAL_FEATURES,
    *SUBTREE_CONTENT_FEATURES,
    *SUBTREE_FUNCTIONAL_FEATURES,
    *SUBTREE_VISUAL_FEATURES,
]

GLOBAL_STRUCTURAL_FEATURES = [
    # global structural
    "n_tree_rem",
    "n_tree_add",
    "n_tree_edit",
]


GLOBAL_VISUAL_FEATURES = [
    # global visual
    "n_sal_tree_rem",
    "n_sal_tree_add",
    "n_sal_tree_edit",
    "n_visible",
    "n_visible_add",
    "n_visible_rem",
]


GLOBAL_FUNCTIONAL_FEATURES = [
    # global functional
    "n_scr",
    "n_scr_add",
    "n_scr_rem",
    "n_scr_err",
    "n_scr_err_add",
    "n_scr_err_rem",
    "n_req",
    "n_req_add",
    "n_req_rem",
]

GLOBAL_FEATURES = [
    *GLOBAL_STRUCTURAL_FEATURES,
    *GLOBAL_VISUAL_FEATURES,
    *GLOBAL_FUNCTIONAL_FEATURES,
]

FEATURES = [*SUBTREE_FEATURES, *GLOBAL_FEATURES]

EDITS_DF_COLUMNS = [
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
    "browser_id",
    "visit_id",
    "id",
    "nodeName",
    "type",
    "attributes",
    "visual_cues",
    "parent_id",
    "t_enter",
    "t_leave",
    "block",
    "domain",
    "top_level_domain",
    "attr_id",
    "attr_class",
    "attr_src",
    "edited",
    "rem_in_r",
    "rem_in_l",
]

EDITS_DF_COLUMNS_LABELED = [
    "new_attr_class",
    "new_attr_id",
    "new_attr_src",
    "new_attr",
    "new_bid",
    "new_id",
    "new_parent",
    "new_te",
    "new_tl",
    "new_vid",
    "new_vc",
    "saliency",
    "party",
    "processed",
    "is_root",
    "prev_bid",
    "prev_vid",
    "prev_id",
    "tag",
    "type",
    "prev_attr",
    "prev_vc",
    "prev_parent",
    "prev_te",
    "prev_tl",
    "block",
    "domain",
    "top_level_domain",
    "prev_attr_id",
    "prev_attr_class",
    "prev_attr_src",
    "edited",
    "removed",
    "added",
    "is_breaking",
    "origin",
    "flipped",
    "issue",
]

NODES_DF_COLUMNS_LABELED = [
    "name",
    "type",
    "value",
    "right_name",
    "left_name",
    "added",
    "removed",
    "origin",
    "flipped",
    "issue",
]

EDGES_DF_COLUMNS_LABELED = [
    "src",
    "dst",
    "type",
    "value",
    "dcount",
    "right_src",
    "right_dst",
    "right_value",
    "left_src",
    "left_dst",
    "left_value",
    "removed",
    "added",
    "count",
    "origin",
    "flipped",
    "issue",
]
