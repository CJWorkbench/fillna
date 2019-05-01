import unittest
import pandas as pd
from pandas.testing import assert_frame_equal
import numpy as np
from fillna import Params, fillna, render, FillWithValue, FillWithPrevious, \
        FillWithNext


class TestParams(unittest.TestCase):
    def test_no_colnames(self):
        p = Params.parse(colnames='', contenttype=0,
                         fillvalue='blah', method=0)
        self.assertEqual(p.colnames, [])

    def test_one_colname(self):
        p = Params.parse(colnames='A', contenttype=0,
                         fillvalue='blah', method=0)
        self.assertEqual(p.colnames, ['A'])

    def test_many_colnames(self):
        p = Params.parse(colnames='A,B', contenttype=0,
                         fillvalue='blah', method=0)
        self.assertEqual(p.colnames, ['A', 'B'])

    def test_fill_with_value(self):
        p = Params.parse(colnames='A', contenttype=0,
                         fillvalue='blah', method=0)
        self.assertIsInstance(p.fill_with, FillWithValue)
        self.assertEqual(p.fill_with.value, 'blah')

    def test_fill_with_previous(self):
        p = Params.parse(colnames='A', contenttype=1,
                         fillvalue='blah', method=0)
        self.assertIsInstance(p.fill_with, FillWithPrevious)

    def test_fill_with_next(self):
        p = Params.parse(colnames='A', contenttype=1, value='blah', method=1)
        self.assertIsInstance(p.fill_with, FillWithNext)


class TestFillna(unittest.TestCase):
    def _test(self, in_table: pd.DataFrame, params: Params,
              expected_out: pd.DataFrame) -> None:
        fillna(in_table, params)  # modifies in_table
        assert_frame_equal(in_table, expected_out)

    def test_no_colnames(self):
        self._test(
            pd.DataFrame({'A': [1, 2]}),
            Params([], FillWithPrevious()),
            pd.DataFrame({'A': [1, 2]})
        )

    def test_invalid_colname(self):
        with self.assertRaises(ValueError, msg='There is no column named B'):
            self._test(
                pd.DataFrame({'A': [1, 2]}),
                Params(['B'], FillWithPrevious()),
                None
            )

    def test_cast_different_columns_differently(self):
        self._test(
            pd.DataFrame({
                'A': ['a', np.nan],
                'B': [1.1, np.nan],
                'C': [1.1, np.nan],
            }),
            Params(['A', 'C'], FillWithValue('v')),
            pd.DataFrame({
                'A': ['a', 'v'],  # stays str
                'B': [1.1, np.nan],  # unmodified
                'C': ['1.1', 'v'],  # converts to str
            })
        )

    def test_empty_value_does_nothing(self):
        # TODO consider actually replacing with empty string? That seems like a
        # valid use case.
        self._test(
            pd.DataFrame({'A': ['a', 'b', np.nan]}),
            Params(['A'], FillWithValue('')),
            pd.DataFrame({'A': ['a', 'b', np.nan]})
        )

    def test_fill_all_empty_column(self):
        self._test(
            pd.DataFrame({'A': [np.nan, np.nan]}, dtype=str),
            Params(['A'], FillWithValue('c')),
            pd.DataFrame({'A': ['c', 'c']})
        )

    def test_str_value(self):
        self._test(
            pd.DataFrame({'A': ['a', 'b', np.nan]}),
            Params(['A'], FillWithValue('c')),
            pd.DataFrame({'A': ['a', 'b', 'c']})
        )

    def test_str_existing_category(self):
        self._test(
            pd.DataFrame({'A': ['a', 'b', np.nan]}, dtype='category'),
            Params(['A'], FillWithValue('a')),
            pd.DataFrame({'A': ['a', 'b', 'a']}, dtype='category')
        )

    def test_str_new_category(self):
        self._test(
            pd.DataFrame({'A': ['a', 'b', np.nan]}, dtype='category'),
            Params(['A'], FillWithValue('c')),
            pd.DataFrame({'A': ['a', 'b', 'c']}, dtype='category')
        )

    def test_str_new_category_but_no_na(self):
        self._test(
            pd.DataFrame({'A': ['a', 'b']}, dtype='category'),
            Params(['A'], FillWithValue('c')),
            pd.DataFrame({'A': ['a', 'b']}, dtype='category')
        )

    def test_float_to_str(self):
        self._test(
            pd.DataFrame({'A': [1.1, 2.2, np.nan]}, dtype=float),
            Params(['A'], FillWithValue('c')),
            pd.DataFrame({'A': ['1.1', '2.2', 'c']}, dtype=str)
        )

    def test_float_to_float(self):
        self._test(
            pd.DataFrame({'A': [1.1, 2.2, np.nan]}, dtype=float),
            Params(['A'], FillWithValue('3.3')),
            pd.DataFrame({'A': [1.1, 2.2, 3.3]})
        )

    def test_fill_with_previous(self):
        self._test(
            pd.DataFrame({'A': [1.1, np.nan, np.nan]}, dtype=float),
            Params(['A'], FillWithPrevious()),
            pd.DataFrame({'A': [1.1, 1.1, 1.1]}, dtype=float)
        )

    def test_fill_with_previous_na_at_start(self):
        self._test(
            pd.DataFrame({'A': [np.nan, 2.2, np.nan]}, dtype=float),
            Params(['A'], FillWithPrevious()),
            pd.DataFrame({'A': [np.nan, 2.2, 2.2]}, dtype=float)
        )

    def test_fill_with_next(self):
        self._test(
            pd.DataFrame({'A': [1.1, np.nan, 3.3]}, dtype=float),
            Params(['A'], FillWithNext()),
            pd.DataFrame({'A': [1.1, 3.3, 3.3]}, dtype=float)
        )

    def test_fill_with_next_na_at_end(self):
        self._test(
            pd.DataFrame({'A': [np.nan, 2.2, np.nan]}, dtype=float),
            Params(['A'], FillWithNext()),
            pd.DataFrame({'A': [2.2, 2.2, np.nan]}, dtype=float)
        )


class TestRender(unittest.TestCase):
    def test_parse_error(self):
        result = render(pd.DataFrame({'A': [1]}), params={})
        self.assertEqual(result, 'Missing colnames')

    def test_run_error(self):
        result = render(pd.DataFrame({'A': [1]}), params={
            'colnames': 'B',
            'fillvalue': 'v',
            'contenttype': 0,
            'method': 0
        })
        self.assertEqual(result, 'There is no column named B')

    def test_integration(self):
        result = render(pd.DataFrame({'A': ['a', np.nan]}), params={
            'colnames': 'A',
            'fillvalue': 'v',
            'contenttype': 0,
            'method': 0
        })
        expected = pd.DataFrame({'A': ['a', 'v']})
        assert_frame_equal(result, expected)


if __name__ == '__main__':
    unittest.main()
