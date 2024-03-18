from datetime import datetime
import json
import sqlite3
import pandas as pd


with sqlite3.connect("./crawl/datadir-full-12-08-2022/crawl-data.sqlite") as con:
    js = pd.read_sql_query(
        "SELECT value from javascript where symbol='window.last_log_event' and operation='set'",
        con,
    )

    

print(js.head())

js = js[js["value"] != "null"]

js[["timestamp", "message", "src", "stack"]] = js.apply(
    lambda x: list(json.loads(x.value).values())[:-1], axis=1, result_type="expand"
)

js.drop(columns=["value"], inplace=True)

js = js[js["message"] != "InternalError: too much recursion"]

errors = js[["message"]].value_counts()

print(errors)

# we should not care about 'InternalError: too much recursion'


print()

# group stuff per interaction

errors_grouped = js.copy()
errors_grouped["interaction"] = None

interaction_periods = []

for _, interaction in interactions.iterrows():

    timestamp = datetime.strptime(interaction.timestamp, "%Y-%m-%dT%H:%M:%SZ")

    if len(interaction_periods):
        interaction_periods[-1][2] = timestamp
    interaction_periods.append([interaction, timestamp, None])

errors_grouped["timestamp"] = js.apply(
    lambda x: datetime.strptime(x.timestamp, "%Y-%m-%dT%H:%M:%S.%fZ"),
    axis=1,
)

max_log_time = errors_grouped["timestamp"].max()
interaction_periods[-1][2] = max_log_time


for interaction, start, end in interaction_periods:

    valid = errors_grouped[
        (errors_grouped["timestamp"] >= start) & (errors_grouped["timestamp"] < end)
    ]

    errors_grouped.loc[
        (errors_grouped["timestamp"] >= start) & (errors_grouped["timestamp"] < end),
        "interaction",
    ] = start

    if len(valid):
        print(
            "action={} - start: {} : end {}".format(
                interaction.type, str(start), str(end)
            )
        )
        for _, row in valid.iterrows():
            print(row.message)
        print()

print(errors_grouped)
