from typing import Any, Dict
import unittest
import pandas as pd
from pandas.testing import assert_frame_equal
import numpy as np
from fillna import migrate_params, render


def P(colnames=[], method="value", value=""):
    return {
        "colnames": colnames,
        "method": method,
        "value": value,
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
            },
        )


class TestRender(unittest.TestCase):
    def _test(
        self, in_table: pd.DataFrame, params: Dict[str, Any], expected_out: pd.DataFrame
    ) -> None:
        result = render(in_table, params)  # modifies in_table
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
        )

    def test_empty_value_does_nothing(self):
        # TODO consider actually replacing with empty string? That seems like a
        # valid use case.
        self._test(
            pd.DataFrame({"A": ["a", "b", np.nan]}),
            P(["A"], "value", ""),
            pd.DataFrame({"A": ["a", "b", np.nan]}),
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
        )

    def test_float_to_float(self):
        self._test(
            pd.DataFrame({"A": [1.1, 2.2, np.nan]}, dtype=float),
            P(["A"], "value", "3.3"),
            pd.DataFrame({"A": [1.1, 2.2, 3.3]}),
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


if __name__ == "__main__":
    unittest.main()
