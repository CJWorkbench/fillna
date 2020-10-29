from typing import Any, Dict, List
import unittest
import pandas as pd
from pandas.testing import assert_frame_equal
import numpy as np
from fillna import migrate_params, render
from cjwmodule.testing.i18n import cjwmodule_i18n_message, i18n_message


def P(colnames=[], method="value", value="", from_colnames=[]):
    return {
        "colnames": colnames,
        "method": method,
        "value": value,
        "from_colnames": from_colnames,
    }


class MigrateParamsTest(unittest.TestCase):
    def test_v0_no_colnames(self):
        self.assertEqual(
            migrate_params(
                {
                    "colnames": "",
                    "contenttype": 0,
                    "fillvalue": "",
                    "method": 0,
                }
            ),
            {
                "colnames": [],
                "method": "value",
                "value": "",
                "from_colnames": [],
            },
        )

    def test_v0_colnames(self):
        self.assertEqual(
            migrate_params(
                {
                    "colnames": "A,B",
                    "contenttype": 0,
                    "fillvalue": "x",
                    "method": 0,
                }
            ),
            {
                "colnames": ["A", "B"],
                "method": "value",
                "value": "x",
                "from_colnames": [],
            },
        )

    def test_v0_topdown(self):
        self.assertEqual(
            migrate_params(
                {
                    "colnames": "",
                    "contenttype": 1,
                    "fillvalue": "",
                    "method": 0,
                }
            ),
            {
                "colnames": [],
                "method": "pad",
                "value": "",
                "from_colnames": [],
            },
        )

    def test_v0_bottomup(self):
        self.assertEqual(
            migrate_params(
                {
                    "colnames": "",
                    "contenttype": 1,
                    "fillvalue": "",
                    "method": 1,
                }
            ),
            {
                "colnames": [],
                "method": "backfill",
                "value": "",
                "from_colnames": [],
            },
        )

    def test_v1(self):
        self.assertEqual(
            migrate_params(
                {
                    "colnames": ["A"],
                    "method": "backfill",
                    "value": "x",
                }
            ),
            {
                "colnames": ["A"],
                "method": "backfill",
                "value": "x",
                "from_colnames": [],
            },
        )

    def test_v2(self):
        self.assertEqual(
            migrate_params(
                {
                    "colnames": ["A"],
                    "method": "backfill",
                    "value": "x",
                }
            ),
            {
                "colnames": ["A"],
                "method": "backfill",
                "value": "x",
                "from_colnames": [],
            },
        )


class TestRender(unittest.TestCase):
    def _test(
        self,
        in_table: pd.DataFrame,
        params: Dict[str, Any],
        expected_out: pd.DataFrame,
        expected_warnings: List = [],
    ) -> None:
        result = render(in_table, params)  # modifies in_table
        if expected_warnings:
            self.assertIsInstance(result, tuple)
            self.assertIsInstance(result[0], pd.DataFrame)
            assert_frame_equal(result[0], expected_out)
            self.assertEqual(result[1], expected_warnings)
        else:
            self.assertIsInstance(result, pd.DataFrame)
            assert_frame_equal(result, expected_out)

    def test_no_colnames(self):
        self._test(pd.DataFrame({"A": [1, 2]}), P([]), pd.DataFrame({"A": [1, 2]}))

    def test_cast_different_columns_differently(self):
        self._test(
            pd.DataFrame(
                {
                    "A": ["a", np.nan],
                    "B": [1.1, np.nan],
                    "C": [1.1, np.nan],
                }
            ),
            P(["A", "C"], method="value", value="v"),
            pd.DataFrame(
                {
                    "A": ["a", "v"],  # stays str
                    "B": [1.1, np.nan],  # unmodified
                    "C": ["1.1", "v"],  # converts to str
                }
            ),
            [
                {
                    "message": i18n_message(
                        "errors.valueNotNumber", {"colname": "C", "value": "v"}
                    )
                }
            ],
        )

    def test_empty_value_is_str_text(self):
        self._test(
            pd.DataFrame({"A": ["a", "b", np.nan]}),
            P(["A"], "value", ""),
            pd.DataFrame({"A": ["a", "b", ""]}),
        )

    def test_empty_value_is_null_number(self):
        self._test(
            pd.DataFrame({"A": [1, 2, np.nan]}),
            P(["A"], "value", ""),
            pd.DataFrame({"A": [1, 2, np.nan]}),
        )

    def test_fill_all_empty_column(self):
        self._test(
            pd.DataFrame({"A": [np.nan, np.nan]}, dtype=str),
            P(["A"], "value", "c"),
            pd.DataFrame({"A": ["c", "c"]}),
        )

    def test_str_value(self):
        self._test(
            pd.DataFrame({"A": ["a", "b", np.nan]}),
            P(["A"], "value", "c"),
            pd.DataFrame({"A": ["a", "b", "c"]}),
        )

    def test_str_existing_category(self):
        self._test(
            pd.DataFrame({"A": ["a", "b", np.nan]}, dtype="category"),
            P(["A"], "value", "a"),
            pd.DataFrame({"A": ["a", "b", "a"]}, dtype="category"),
        )

    def test_str_new_category(self):
        self._test(
            pd.DataFrame({"A": ["a", "b", np.nan]}, dtype="category"),
            P(["A"], "value", "c"),
            pd.DataFrame({"A": ["a", "b", "c"]}, dtype="category"),
        )

    def test_str_new_category_but_no_na(self):
        self._test(
            pd.DataFrame({"A": ["a", "b"]}, dtype="category"),
            P(["A"], "value", "c"),
            pd.DataFrame({"A": ["a", "b"]}, dtype="category"),
        )

    def test_float_to_str(self):
        self._test(
            pd.DataFrame({"A": [1.1, 2.2, np.nan]}, dtype=float),
            P(["A"], "value", "c"),
            pd.DataFrame({"A": ["1.1", "2.2", "c"]}, dtype=str),
            [
                {
                    "message": i18n_message(
                        "errors.valueNotNumber", {"colname": "A", "value": "c"}
                    )
                }
            ],
        )

    def test_timestamp_to_str(self):
        self._test(
            pd.DataFrame(
                {"A": pd.Series(["2020-01-01", pd.NaT], dtype="datetime64[ns]")}
            ),
            P(["A"], "value", "c"),
            pd.DataFrame({"A": ["2020-01-01", "c"]}, dtype=str),
            [
                {
                    "message": i18n_message(
                        "errors.valueNotTimestamp", {"colname": "A", "value": "c"}
                    )
                }
            ],
        )

    def test_float_to_float(self):
        # This also covers "int to float", because Pandas int+null columns are float
        self._test(
            pd.DataFrame({"A": [1.1, 2.2, np.nan]}, dtype=float),
            P(["A"], "value", "3.3"),
            pd.DataFrame({"A": [1.1, 2.2, 3.3]}),
        )

    def test_timestamp_to_timestamp(self):
        self._test(
            pd.DataFrame(
                {"A": pd.Series(["2020-01-01", pd.NaT], dtype="datetime64[ns]")}
            ),
            P(["A"], "value", "2020-10-29T11:21"),
            pd.DataFrame(
                {"A": ["2020-01-01", "2020-10-29T11:21"]}, dtype="datetime64[ns]"
            ),
        )

    def test_timestamp_to_timestamp_value_has_timezone(self):
        self._test(
            pd.DataFrame(
                {"A": pd.Series(["2020-01-01", pd.NaT], dtype="datetime64[ns]")}
            ),
            P(["A"], "value", "2020-10-29T11:21+01:00"),
            pd.DataFrame(
                {"A": ["2020-01-01", "2020-10-29T10:21"]}, dtype="datetime64[ns]"
            ),
        )

    def test_fill_with_previous(self):
        self._test(
            pd.DataFrame({"A": [1.1, np.nan, np.nan]}, dtype=float),
            P(["A"], "pad"),
            pd.DataFrame({"A": [1.1, 1.1, 1.1]}, dtype=float),
        )

    def test_fill_with_previous_na_at_start(self):
        self._test(
            pd.DataFrame({"A": [np.nan, 2.2, np.nan]}, dtype=float),
            P(["A"], "pad"),
            pd.DataFrame({"A": [np.nan, 2.2, 2.2]}, dtype=float),
        )

    def test_fill_with_next(self):
        self._test(
            pd.DataFrame({"A": [1.1, np.nan, 3.3]}, dtype=float),
            P(["A"], "backfill"),
            pd.DataFrame({"A": [1.1, 3.3, 3.3]}, dtype=float),
        )

    def test_fill_with_next_na_at_end(self):
        self._test(
            pd.DataFrame({"A": [np.nan, 2.2, np.nan]}, dtype=float),
            P(["A"], "backfill"),
            pd.DataFrame({"A": [2.2, 2.2, np.nan]}, dtype=float),
        )

    def test_fill_with_columns_empty(self):
        self._test(
            pd.DataFrame({"A": [1, np.nan, 2]}),
            P(["A"], "columns", from_colnames=[]),
            pd.DataFrame({"A": [1, np.nan, 2]}),
        )

    def test_fill_with_columns_self(self):
        self._test(
            pd.DataFrame({"A": [1, np.nan, 2]}),
            P(["A"], "columns", from_colnames=["A"]),
            pd.DataFrame({"A": [1, np.nan, 2]}),
        )

    def test_fill_with_columns_multiple(self):
        self._test(
            pd.DataFrame(
                {
                    "A": [1, np.nan, np.nan, np.nan],
                    "B": [2, 2, np.nan, np.nan],
                    "C": [3, 3, 3, np.nan],
                }
            ),
            P(["A"], "columns", from_colnames=["B", "C"]),
            pd.DataFrame(
                {
                    "A": [1, 2, 3, np.nan],
                    "B": [2, 2, np.nan, np.nan],
                    "C": [3, 3, 3, np.nan],
                }
            ),
        )

    def test_fill_categorical_with_str_column(self):
        self._test(
            pd.DataFrame(
                {
                    "A": pd.Series(["a", None, None], dtype="category"),
                    "B": ["b", "b", None],
                }
            ),
            P(["A"], "columns", from_colnames=["B"]),
            pd.DataFrame(
                {
                    "A": ["a", "b", None],
                    "B": ["b", "b", None],
                }
            ),
        )

    def test_fill_str_with_categorical_column(self):
        self._test(
            pd.DataFrame(
                {
                    "A": ["a", None, None],
                    "B": pd.Series(["b", "b", None], dtype="category"),
                }
            ),
            P(["A"], "columns", from_colnames=["B"]),
            pd.DataFrame(
                {
                    "A": ["a", "b", None],
                    "B": pd.Series(["b", "b", None], dtype="category"),
                }
            ),
        )

    def test_fill_categorical_with_categorical_column(self):
        self._test(
            pd.DataFrame(
                {
                    "A": pd.Series(["a", None, None], dtype="category"),
                    "B": pd.Series(["b", "b", None], dtype="category"),
                }
            ),
            P(["A"], "columns", from_colnames=["B"]),
            pd.DataFrame(
                {
                    "A": pd.Series(["a", "b", None], dtype="category"),
                    "B": pd.Series(["b", "b", None], dtype="category"),
                }
            ),
        )

    def test_fill_with_columns_multiple_change_input_to_text(self):
        self._test(
            pd.DataFrame(
                {
                    "A": [1, np.nan, np.nan, np.nan],
                    "B": ["b", "b", None, None],
                    "C": ["c", "c", "c", None],
                }
            ),
            P(["A"], "columns", from_colnames=["B", "C"]),
            pd.DataFrame(
                {
                    "A": ["1.0", "b", "c", None],  # [sic] -- TODO format numbers
                    "B": ["b", "b", None, None],
                    "C": ["c", "c", "c", None],
                }
            ),
            [
                {
                    "message": i18n_message(
                        "errors.valueColumnsWrongType",
                        {"colname": "A", "value_colnames": []},
                    ),
                }
            ],
        )

    def test_fill_with_columns_multiple_change_from_columns_to_text(self):
        self._test(
            pd.DataFrame(
                {
                    "A": ["a", None, None, None],
                    "B": [2, 2, np.nan, np.nan],
                    "C": pd.Series(
                        ["2003-03-03", "2003-03-03", "2003-03-03", None],
                        dtype="datetime64[ns]",
                    ),
                }
            ),
            P(["A"], "columns", from_colnames=["B", "C"]),
            pd.DataFrame(
                {
                    "A": ["a", "2.0", "2003-03-03", None],
                    "B": [2, 2, np.nan, np.nan],
                    "C": pd.Series(
                        ["2003-03-03", "2003-03-03", "2003-03-03", None],
                        dtype="datetime64[ns]",
                    ),
                }
            ),
            [
                {
                    "message": i18n_message(
                        "errors.valueColumnsWrongType",
                        {"colname": "A", "value_colnames": ["B", "C"]},
                    ),
                }
            ],
        )

    def test_fill_with_columns_change_input_and_from_columns_to_text(self):
        self._test(
            pd.DataFrame(
                {
                    "A": [1, np.nan, np.nan, np.nan],
                    "B": [2, 2, np.nan, np.nan],
                    "C": pd.Series(
                        ["2003-03-03", "2003-03-03", "2003-03-03", None],
                        dtype="datetime64[ns]",
                    ),
                }
            ),
            P(["A"], "columns", from_colnames=["B", "C"]),
            pd.DataFrame(
                {
                    "A": ["1.0", "2.0", "2003-03-03", None],
                    "B": [2, 2, np.nan, np.nan],
                    "C": pd.Series(
                        ["2003-03-03", "2003-03-03", "2003-03-03", None],
                        dtype="datetime64[ns]",
                    ),
                }
            ),
            [
                {
                    "message": i18n_message(
                        "errors.valueColumnsWrongType",
                        {"colname": "A", "value_colnames": ["B", "C"]},
                    ),
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
