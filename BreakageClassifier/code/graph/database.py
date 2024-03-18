from datetime import datetime
from pathlib import Path
from typing import Optional
import sqlite3

import pandas as pd

from BreakageClassifier.code.graph.constants import JS_ELEM_SYMBOLS, JS_ERROR_SYMBOLS


class Database:

    """Database class to perform all functions associated with the OpenWPM crawl DB."""

    def __init__(self, database_filename: Path, log_filename: Path = None):
        self.database_filename = database_filename
        self.log_filename = log_filename
        self.conn: Optional[sqlite3.Connection] = None
        self.log_df: Optional[pd.DataFrame] = None

    def __enter__(self):
        self.conn = sqlite3.connect(str(self.database_filename.resolve()))

        if self.log_filename:
            self.log_df = pd.read_csv(str(self.log_filename.resolve()))
            if "Unnamed: 0" in self.log_df.columns:
                self.log_df.drop(columns=["Unnamed: 0"], inplace=True)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn is not None:
            self.conn.close()
        self.conn = None

    def website_from_visit_id(self, visit_id):
        """
        Function to get relevant table data for a particular website.
        This data is ued to build graphs.

        Args:
            visit_id: Visit ID of the website we want.
        Returns:
            df_requests: DataFrame representation of requests table in OpenWPM.
            df_responses: DataFrame representation of responses table in OpenWPM.
            df_redirects: DataFrame representation of redirects table in OpenWPM.
            call_stacks: DataFrame representation of call_stacks table in OpenWPM.
            javascript: DataFrame representation of javascript table in OpenWPM.
        """

        if self.conn is None:
            raise sqlite3.ProgrammingError("Database not open")

        df_http_requests = pd.read_sql_query(
            "SELECT visit_id, request_id, "
            "url, headers, top_level_url, resource_type, "
            f"time_stamp, post_body, post_body_raw from http_requests where {visit_id} = visit_id",
            self.conn,
        )
        df_http_responses = pd.read_sql_query(
            "SELECT visit_id, request_id, "
            "url, headers, response_status, time_stamp, content_hash "
            f" from http_responses where {visit_id} = visit_id",
            self.conn,
        )
        df_http_redirects = pd.read_sql_query(
            "SELECT visit_id, old_request_id, "
            "old_request_url, new_request_url, response_status, "
            f"headers, time_stamp from http_redirects where {visit_id} = visit_id",
            self.conn,
        )
        call_stacks = pd.read_sql_query(
            f"SELECT visit_id, request_id, call_stack from callstacks where {visit_id} = visit_id",
            self.conn,
        )
        javascript = pd.read_sql_query(
            "SELECT visit_id, script_url, script_line, script_loc_eval, top_level_url, document_url, symbol, call_stack, operation,"
            f" arguments, attributes, value, time_stamp from javascript where {visit_id} = visit_id",
            self.conn,
        )

        return (
            df_http_requests,
            df_http_responses,
            df_http_redirects,
            call_stacks,
            javascript,
        )

    def sites_visits(self):
        """
        Function to get site visit table data.

        Returns:
            DataFrame representation of the site_visits table for successfully crawled sites.
        """

        def expand_visits(group: pd.DataFrame):
            row = {
                "visit_id_b": None,
                "browser_id_b": None,
                "visit_id_u": None,
                "browser_id_u": None,
                "visit_id_a": None,
                "browser_id_a": None,
                "error": True,
            }

            if len(group) >= 3:
                # find the last 3 consecutive crawls
                # we can assume here that they are sorted by site rank
                # the crawl order is after, before, unfiltered

                state = 0
                states = ["a", "b", "u"]

                visits = []

                i = 0

                while i < len(group):
                    r = group.iloc[i]

                    if not r.filterlist or isinstance(r.filterlist, float):
                        r.filterlist = "unfiltered"

                    if r.filterlist[0] == states[state]:
                        # we passed
                        visits.append((r.visit_id, r.browser_id))
                        state += 1
                        i += 1

                    else:
                        if state == 0:
                            i += 1
                        else:
                            state = 0
                            visits = []

                    if state == 3:
                        break

                row["error"] = len(visits) != 3

                if not row["error"]:
                    for i, (visit_id, browser_id) in enumerate(visits):
                        row[f"visit_id_{states[i]}"] = visit_id
                        row[f"browser_id_{states[i]}"] = browser_id

            row["issue_id"] = group.iloc[0].id
            row["site_url"] = group.iloc[0].site_url

            return pd.Series(row)

        df_successful_sites = pd.read_sql(
            """select distinct s.visit_id as visit_id, s.browser_id as browser_id, site_rank, s.site_url as site_url 
            from site_visits s
            join crawl_history c on s.visit_id = c.visit_id and s.browser_id = c.browser_id
            where c.command='FinalizeCommand' and command_status='ok' and site_rank != -1
            """,
            self.conn,
        )
        # only site ranks that are integers 
        df_successful_sites = df_successful_sites[df_successful_sites["site_rank"].apply(lambda x: isinstance(x, int))]
        
        df_successful_sites = (
            df_successful_sites.sort_values(by="site_rank")
            .merge(self.log_df, how="inner", on="site_rank")
            .groupby("site_url")
        )

        df_successful_sites = df_successful_sites.apply(expand_visits)

        df_successful_sites = df_successful_sites[~df_successful_sites["error"]]

        return df_successful_sites

    def get_dom(self, visit_ids=None):
        if visit_ids is None:
            return pd.read_sql(f"select * from dom_nodes", self.conn)
        else:
            visit_ids = [str(i) for i in visit_ids]

            return pd.read_sql(
                f"select * from dom_nodes where visit_id in ({','.join(visit_ids)})",
                self.conn,
            )
            
    def get_commands(self, visit_id=None):
        
        return pd.read_sql(
            f"select command, duration from crawl_history where visit_id={visit_id}",
            self.conn,
        )

    def get_dom_from_visit_id(self, visit_id: str):
        return pd.read_sql(
            f"select * from dom_nodes where visit_id={visit_id}", self.conn
        )

    def get_requests(self):
        return pd.read_sql(
            "select visit_id, request_id, url, method, frame_id, response_status, time_stamp from http_responses",
            self.conn,
        )

    def get_http_responses(self, visit_id: str):
        return pd.read_sql(
            f"select request_id, url, method, frame_id, response_status, time_stamp from http_responses where visit_id={visit_id}",
            self.conn,
        )

    def get_request_response(self, visit_ids=None):
        visits_where = ""

        if visit_ids is not None:
            visit_ids = [str(i) for i in visit_ids]

            visits_where = f"where http_requests.visit_id in ({','.join(visit_ids)})"
        return pd.read_sql(
            f"""select http_requests.visit_id as visit_id, http_requests.parent_frame_id as parent_frame_id,  http_responses.headers as resp_headers, http_requests.request_id as request_id, http_requests.url as url, http_requests.top_level_url as top_level_url, resource_type, response_status from 
            http_responses 
            join http_requests on http_responses.request_id = http_requests.request_id AND http_responses.visit_id = http_requests.visit_id
            {visits_where}
            """,
            self.conn,
        )

    def get_all_callstacks(self, visit_ids=None):
        if visit_ids is None:
            return pd.read_sql(
                f"select visit_id, request_id, call_stack from callstacks",
                self.conn,
            )
        else:
            visit_ids = [str(i) for i in visit_ids]
            return pd.read_sql(
                f"select visit_id, request_id, call_stack from callstacks where visit_id in ({','.join(visit_ids)})",
                self.conn,
            )

    def get_callstacks(self, visit_id: str):
        return pd.read_sql(
            f"select request_id, call_stack from callstacks where visit_id={visit_id}",
            self.conn,
        )

    def get_instrumented_errors(self, visit_id: str):
        errors: pd.DataFrame = pd.read_sql(
            f"SELECT * from javascript where (visit_id ={visit_id} AND (symbol = 'window.console.warn' OR symbol = 'window.console.error') AND operation='call')",
            self.conn,
        )
        return errors

    def get_interaction_logs(self, visit_id: str):
        js: pd.DataFrame = pd.read_sql(
            f"SELECT * from js_errors where visit_id ={visit_id}", self.conn
        )

        js = js[js["message"] != "InternalError: too much recursion"]

        interactions: pd.DataFrame = pd.read_sql(
            "SELECT * from interaction_annotations order by timestamp",
            self.conn,
        )

        interactions["timestamp"] = interactions.apply(
            lambda x: datetime.strptime(x.timestamp, "%Y-%m-%dT%H:%M:%SZ"),
            axis=1,
        )

        return Database._parse_events_for_interactions(interactions, js)

    def get_all_javascript_events(
        self, visit_ids=None, symbols=JS_ELEM_SYMBOLS + JS_ERROR_SYMBOLS, 
    ):
        visits_where = ""

        if visit_ids is not None:
            visit_ids = [str(i) for i in visit_ids]

            visits_where = f"AND visit_id in ({','.join(visit_ids)})"

        if symbols is not None:
            symbols_s = ",".join(f'"{s}"' for s in symbols)
        
            symbols_where = f"symbol IN ({symbols_s})"
            
        else:
            symbols_where = "1=1"

        javascript: pd.DataFrame = pd.read_sql(
            f"""
                SELECT * from javascript 
                where 
                    {symbols_where}
                    {visits_where}
                ORDER by time_stamp
            """,
            self.conn,
        )

        return javascript

    def get_javascript_events(
        self,
        visit_id: str,
        symbols=JS_ELEM_SYMBOLS + JS_ERROR_SYMBOLS,
    ):
        symbols_s = ",".join(f'"{s}"' for s in symbols)

        javascript: pd.DataFrame = pd.read_sql(
            f"""
                SELECT * from javascript 
                where 
                    visit_id={visit_id}
                    AND symbol IN ({symbols_s}) 
                ORDER by time_stamp
            """,
            self.conn,
        )

        return javascript
    
    def get_runtime(self):
        return pd.read_sql(
            f"SELECT visit_id, command, duration from crawl_history", self.conn
        )

    def _get_interaction_annotations(self, visit_id: str):
        interactions: pd.DataFrame = pd.read_sql(
            f"SELECT * from interaction_annotations where visit_id={visit_id} order by timestamp",
            self.conn,
        )

        if not interactions.empty:
            interactions["timestamp"] = interactions.apply(
                lambda x: datetime.strptime(x.timestamp, "%Y-%m-%dT%H:%M:%SZ"),
                axis=1,
            )

        else:
            interactions = pd.DataFrame(
                columns=interactions.columns.tolist() + ["timestamp"]
            )

        return interactions

    def get_interaction_logs_all(self, visit_id: str, javascript, http_responses):
        interactions = self._get_interaction_annotations(visit_id)
        _javascript = Database._parse_events_for_interactions(interactions, javascript)
        _http_responses = Database._parse_events_for_interactions(
            interactions, http_responses
        )

        return interactions, _javascript, _http_responses

    @staticmethod
    def _parse_events_for_interactions(
        interactions: pd.DataFrame, events: pd.DataFrame
    ):
        _events = events.copy()

        # this assumes that events has a timestamp field
        assert (
            "time_stamp" in _events.columns or "timestamp" in _events.columns
        ), f"No timestamp for table: {_events.columns}"

        timstamp_column_name = (
            "time_stamp" if "time_stamp" in _events.columns else "timestamp"
        )

        _events["interaction"] = None

        if not len(_events):
            return _events

        interaction_periods = []

        for _, interaction in interactions.iterrows():
            timestamp = interaction.timestamp
            if len(interaction_periods):
                interaction_periods[-1][2] = timestamp
            interaction_periods.append([interaction, timestamp, None])

        _events["timestamp"] = _events.apply(
            lambda x: datetime.strptime(
                x[timstamp_column_name], "%Y-%m-%dT%H:%M:%S.%fZ"
            ),
            axis=1,
        )

        if interactions.empty:
            return _events

        max_log_time = _events["timestamp"].max()

        interaction_periods[-1][2] = max_log_time

        for interaction, start, end in interaction_periods:
            _events.loc[
                (_events["timestamp"] >= start) & (_events["timestamp"] <= end),
                "interaction",
            ] = start

        return _events
