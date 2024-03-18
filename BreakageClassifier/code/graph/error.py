import pandas as pd
import uuid

from .utils import (
    update_children_t,
    update_t_enter_leave_after_insert_subtree,
)
from .constants import ERROR_TYPE, INTERACTION_TYPE

# DEF: node Type 10 is an interaction

ERROR_CATEGORIES = {
    "reference-error": {
        "ReferenceError",
    },
    "eval-error": {
        "EvalError",
    },
    "internal-error": {
        "InternalError",
    },
    "type-error": {
        "TypeError",
    },
}


def _interaction_to_node(interaction: pd.Series):

    return (
        interaction.browser_id,
        interaction.visit_id,
        str(interaction.timestamp),
        interaction.type + "-action",
        INTERACTION_TYPE,
        "[{'key':'id', "
        + f"'value':'{interaction.type}-{interaction.node_id}'"
        + "},]",
        "{}",
        interaction.node_id,
        None,
        None,
        None,
        interaction.block_id,
        interaction.timestamp,
    )


def _error_to_node(error: pd.Series):

    error_cat = "other-error"

    for key in ERROR_CATEGORIES:
        if any(word in error.message for word in ERROR_CATEGORIES[key]):
            error_cat = key
            break

    return (
        error.browser_id,
        error.visit_id,
        str(uuid.uuid4()),
        error_cat,
        ERROR_TYPE,
        "[{'key':'id', " + f"'value':'{error.message}'" + "},]",
        "{}",
        None,
        None,
        None,
        None,
        None,
    )


def augment_dom_with_interactions(dom: pd.DataFrame, interactions: pd.DataFrame):
    # make new virtual nodes

    # interaction_nodes = pd.DataFrame(columns=list(dom.columns) + ["timestamp"])
    # interaction_nodes[list(dom.columns) + ["timestamp"]] = interactions.apply(
    #     _interaction_to_node, result_type="expand", axis=1
    # )

    interaction_nodes = []
    # error_nodes = pd.DataFrame(columns=dom.columns)
    columns = list(dom.columns) + ["timestamp"]

    for _, interaction in interactions.iterrows():
        interaction_node = {
            columns[i]: v for i, v in enumerate(_interaction_to_node(interaction))
        }

        # _en = pd.DataFrame(columns=dom.columns)

        _this_interaction_nodes = dom[dom["id"] == interaction_node["parent_id"]]

        if len(_this_interaction_nodes):
            parent_node = _this_interaction_nodes.iloc[0]
            interaction_node["t_enter"] = parent_node["t_leave"]
            interaction_node["t_leave"] = parent_node["t_leave"] + 1


            dom = update_t_enter_leave_after_insert_subtree(
                dom, parent_node, interaction_node["t_leave"]
            )

        else:
            parent_node = None
            continue

            # if len(interaction_errors):
            #     _en[dom.columns] = interaction_errors[
            #         interaction_errors["interaction"] == interaction.timestamp
            #     ].apply(_error_to_node, result_type="expand", axis=1)

            #    interaction_nodes[-1]["t_enter"] = parent_node["t_leave"]

            #     _en, interaction_nodes[-1]["t_leave"] = update_children_t(
            #         interaction_nodes[-1], _en
            #     )

            # error_nodes = pd.concat([error_nodes, _en])

    interaction_nodes = pd.DataFrame(interaction_nodes)

    if len(interaction_nodes):
        interaction_nodes.drop(columns=["timestamp"], inplace=True)
        dom = pd.concat([dom, interaction_nodes])

    return dom


if __name__ == "__main__":

    from database import Database

    #### TESTING augment_dom_with_interactions

    # setup
    df_dom = pd.read_csv("error_test_dom.csv")
    df_interactions = pd.read_csv("error_test_interactions.csv")
    df_errors = pd.read_csv("error_test_errors.csv")

    df_interactions, df_errors = Database._parse_interactions_errors_from_db(
        df_interactions, df_errors
    )

    df = augment_dom_with_interactions(df_dom, df_errors, df_interactions)



