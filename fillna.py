import pandas as pd
from typing import List


class FillWith:
    """Abstract class describing how to fill missing values."""

    def run(series: pd.Series) -> pd.Series:
        """Return a new `series` with NA values filled in."""
        raise NotImplementedError


class FillWithValue(FillWith):
    """
    Operation that fills missing values with a provided value.

    The provided value is given as ``str``; `apply()` will attempt to convert
    it to the series type, if possible.
    """

    def __init__(self, value: str):
        self.value = value

    def _convert_to_str_and_fill(self, series: pd.Series) -> pd.Series:
        """Convert all values to str and fill in missing ones."""
        # Usually the input will _already_ be str. But let's play it safe and
        # convert anyway.
        series2 = series.astype(str)
        series2[series.isna()] = self.value
        return series2

    def run(self, series: pd.Series) -> pd.Series:
        if not self.value:
            # TODO consider letting the user replace NA with '', if wanted
            return series

        if hasattr(series, 'cat'):
            # Workbench guarantees categories are always str
            if self.value not in series.cat.categories:
                series = series.cat.add_categories([self.value])

            series = series.fillna(self.value)
        elif pd.api.types.is_numeric_dtype(series):
            try:
                numeric_value = pd.to_numeric(self.value)
                series = series.fillna(numeric_value)
            except ValueError:
                series = self._convert_to_str_and_fill(series)

        else:
            series = self._convert_to_str_and_fill(series)

        return series


class FillWithPrevious(FillWith):
    """Operation that fills missing values with previous ones in the Series."""

    def run(self, series: pd.Series) -> pd.Series:
        return series.fillna(method='pad')


class FillWithNext(FillWith):
    """Operation that fills missing values with next ones in the Series."""

    def run(self, series: pd.Series) -> pd.Series:
        return series.fillna(method='backfill')


class Params:
    def __init__(self, colnames: List[str], fill_with: FillWith):
        self.colnames = colnames
        self.fill_with = fill_with

    @staticmethod
    def parse(**kwargs) -> 'Params':
        """Parse params, or raise ValueError."""
        if 'colnames' not in kwargs:
            raise ValueError('Missing colnames')
        colnames = list([c for c in str(kwargs['colnames']).split(',') if c])

        if 'contenttype' not in kwargs:
            raise ValueError('Missing contenttype')
        contenttype = str(kwargs['contenttype'])

        if contenttype == '0':
            if 'fillvalue' not in kwargs:
                raise ValueError('Missing fillvalue')
            fill_with = FillWithValue(str(kwargs['fillvalue']))
        elif contenttype == '1':
            if 'method' not in kwargs:
                raise ValueError('Missing method')
            method = str(kwargs['method'])
            if method == '0':
                fill_with = FillWithPrevious()
            elif method == '1':
                fill_with = FillWithNext()
            else:
                raise ValueError('Invalid method index {method}')
        else:
            raise ValueError('Invalid contenttype index {contenttype}')

        return Params(colnames, fill_with)


def fillna(table: pd.DataFrame, params: Params) -> None:
    """Modify table, or raise ValueError."""
    for colname in params.colnames:
        if colname not in table.columns:
            # error
            raise ValueError(f'There is no column named {colname}')

        series = table[colname]
        series2 = params.fill_with.run(series)
        table[colname] = series2


def render(table, params):
    try:
        parsed_params = Params.parse(**params)
    except ValueError as err:
        return str(err)

    try:
        fillna(table, parsed_params)
    except ValueError as err:
        return str(err)

    return table
