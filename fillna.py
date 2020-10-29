from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List
import pandas as pd


class FillWith(ABC):
    """Abstract class describing how to fill missing values."""

    @abstractmethod
    def run(series: pd.Series) -> pd.Series:
        """Return a new `series` with NA values filled in."""

    @classmethod
    def parse(cls, method: str, value: str) -> FillWith:
        if method == "value":
            return FillValue(value)
        elif method == "pad":
            return FillPad()
        elif method == "backfill":
            return FillBackfill()
        else:
            raise ValueError(f"Invalid method {method}")


class FillValue(FillWith):
    """
    Operation that fills missing values with a provided value.

    The provided value is given as ``str``; `apply()` will attempt to convert
    it to the series type, if possible.
    """

    def __init__(self, value: str):
        self.value = value

    def _convert_to_str_and_fill(self, series: pd.Series) -> pd.Series:
        """Convert all values to str and fill in missing ones."""
        # FIXME add a suggestion to quick-fix? Or better yet, find a nice way
        # to prompt the user and fail, instead of converting automatically.

        # Usually the input will _already_ be str. But let's play it safe and
        # convert anyway.
        series2 = series.astype(str)
        series2[series.isna()] = self.value
        return series2

    def run(self, series: pd.Series) -> pd.Series:
        if not series.isnull().any():
            return series

        if hasattr(series, "cat"):
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


class FillPad(FillWith):
    """Operation that fills missing values with previous ones in the Series."""

    def run(self, series: pd.Series) -> pd.Series:
        return series.fillna(method="pad")


class FillBackfill(FillWith):
    """Operation that fills missing values with next ones in the Series."""

    def run(self, series: pd.Series) -> pd.Series:
        return series.fillna(method="backfill")


def fillna(table: pd.DataFrame, colnames: List[str], fill_with: FillWith) -> None:
    for colname in colnames:
        series = table[colname]
        series2 = fill_with.run(series)
        table[colname] = series2


def render(table, params):
    fill_with = FillWith.parse(params["method"], params["value"])
    fillna(table, params["colnames"], fill_with)
    return table


def _migrate_params_v0_to_v1(params):
    """
    v0: 'colnames' is comma-separated str; menus are integers
    'contenttype' (0=value, 1=show another menu) and
    'method' (0=pad, 1=backfill). value is 'fillvalue'.

    v1: 'colnames' is List[str]; menu options are value|pad|backfill.
    value is 'value'.

    (These all bring params in line with pandas.DataFrame.fillna.)
    """
    method = {
        (0, 0): "value",
        (0, 1): "value",
        (1, 0): "pad",
        (1, 1): "backfill",
    }[(params["contenttype"], params["method"])]
    return {
        "colnames": [c for c in params["colnames"].split(",") if c],
        "method": method,
        "value": params["fillvalue"],
    }


def migrate_params(params):
    if "contenttype" in params:
        params = _migrate_params_v0_to_v1(params)
    return params
