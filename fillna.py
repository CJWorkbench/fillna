from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import datetime
import dateutil
from typing import List
from cjwmodule.i18n import trans, I18nMessage

import pandas as pd


def _warn_converted_to_text_because_value_not_timestamp(colname: str, value: str):
    return {
        "message": trans(
            "errors.valueNotTimestamp",
            "Column “{colname}” was converted to Text because the given value is not a Timestamp. "
            "Try entering a value that looks like “2020-01-10” or “2020-01-10T13:11”.",
            # This message doesn't take a "value" argument, but let's
            # store arguments in case we change the message in the future.
            {"colname": colname, "value": value},
        )
    }


def _warn_converted_to_text_because_value_not_number(colname: str, value: str):
    return {
        "message": trans(
            "errors.valueNotNumber",
            "Column “{colname}” was converted to Text because the given value is not a Number. "
            "Try entering a value that looks like “1234” or “12.31242”.",
            # This message doesn't take a "value" argument, but let's
            # store arguments in case we change the message in the future.
            {"colname": colname, "value": value},
        )
    }


def _warn_converted_to_text_because_types_conflict(
    colname: str, value_colnames: List[str]
):
    return {
        "message": trans(
            "errors.valueColumnsWrongType",
            "Values in column “{colname}” were converted to Text because "
            "the chosen columns have different types.",
            # This message doesn't take a "value_colnames" argument, but let's
            # store arguments in case we change the message in the future.
            {"colname": colname, "value_colnames": value_colnames},
        )
    }


def _workbench_type(series: pd.Series) -> Literal["text", "number", "timestamp"]:
    if pd.api.types.is_numeric_dtype(series):
        return "number"
    elif pd.api.types.is_datetime64_dtype(series):
        return "timestamp"
    else:
        return "text"


def _convert_to_str(series: pd.Series) -> pd.Series:
    ret = series.astype(str)
    ret[series.isna()] = None
    return ret


class FillWith(ABC):
    """Abstract class describing how to fill missing values."""

    @abstractmethod
    def run(series: pd.Series) -> Tuple[pd.Series, List]:
        """Return a new `series` with NA values filled in."""

    @classmethod
    def parse(cls, method: str, value: str, from_columns: List[Series]) -> FillWith:
        if method == "value":
            return FillValue(value)
        elif method == "pad":
            return FillPad()
        elif method == "backfill":
            return FillBackfill()
        elif method == "columns":
            return FillWithColumns(from_columns)
        else:
            raise ValueError(f"Invalid method {method}")


class FillValue(FillWith):
    """Operation that fills missing values with a provided value.

    The provided value is given as ``str``; `apply()` will attempt to convert
    it to the series type, if possible.
    """

    def __init__(self, value: str):
        self.value = value

    def run(self, series: pd.Series):
        if not series.isnull().any():
            # There are no nulls. No-op.
            return series, []

        if self.value == "" and _workbench_type(series) in {"number", "timestamp"}:
            # "" means null for timestamps and numbers
            return series, []

        value = self.value
        warnings = []

        # Try to convert `value` to series type. If we fail, convert `series`
        # to str and add a warning.
        if _workbench_type(series) == "number":
            try:
                value = float(self.value)
            except ValueError:
                warnings.append(
                    _warn_converted_to_text_because_value_not_number(series.name, value)
                )
                series = _convert_to_str(series)
        elif _workbench_type(series) == "timestamp":
            try:
                value = dateutil.parser.isoparse(value)
                # TODO test nixing timezone
                if value.tzinfo:
                    value = value.astimezone(datetime.timezone.utc).replace(tzinfo=None)
            except ValueError:
                warnings.append(
                    _warn_converted_to_text_because_value_not_timestamp(
                        series.name, value
                    )
                )
                series = _convert_to_str(series)

        # category (of text) series: value is str; make sure we can fillna() with it
        if hasattr(series, "cat") and self.value not in series.cat.categories:
            series = series.cat.add_categories([self.value])

        return series.fillna(value), warnings


class FillPad(FillWith):
    """Operation that fills missing values with previous ones in the Series."""

    def run(self, series: pd.Series):
        return series.fillna(method="pad"), []


class FillBackfill(FillWith):
    """Operation that fills missing values with next ones in the Series."""

    def run(self, series: pd.Series):
        return series.fillna(method="backfill"), []


class FillWithColumns(FillWith):
    """Operation that fills missing values using other columns' values."""

    def __init__(self, from_columns: List[pd.Series]):
        self.from_columns = from_columns

    def run(self, series: pd.Series):
        warnings = []

        output_is_categorical = hasattr(series, "cat") and all(
            hasattr(c, "cat") for c in self.from_columns
        )

        if hasattr(series, "cat"):
            series = _convert_to_str(series)

        best_type = _workbench_type(series)
        for from_column in self.from_columns:
            if _workbench_type(from_column) != best_type:
                # Unhappy path: convert everything to text
                if best_type != "text":
                    series = _convert_to_str(series)

                from_columns = [
                    col if _workbench_type(col) == "text" else _convert_to_str(col)
                    for col in self.from_columns
                ]
                warnings.append(
                    _warn_converted_to_text_because_types_conflict(
                        series.name,
                        [
                            col.name
                            for col, orig in zip(from_columns, self.from_columns)
                            if col is not orig
                        ],
                    )
                )

                break
        else:
            from_columns = self.from_columns

        ret = series.copy()
        for from_column in from_columns:
            if hasattr(from_column, "cat"):
                from_column = _convert_to_str(from_column)
            ret.fillna(from_column, inplace=True)

        if output_is_categorical:
            ret = ret.astype("category")

        return ret, warnings


def fillna(table: pd.DataFrame, colnames: List[str], fill_with: FillWith) -> None:
    warnings = []
    for colname in colnames:
        series = table[colname]
        series2, series_warnings = fill_with.run(series)
        warnings.extend(series_warnings)
        table[colname] = series2
    return warnings


def render(table, params):
    fill_with = FillWith.parse(
        params["method"], params["value"], [table[c] for c in params["from_colnames"]]
    )
    warnings = fillna(table, params["colnames"], fill_with)
    if warnings:
        return table, warnings
    else:
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


def _migrate_params_v1_to_v2(params):
    """v1: No "from_colnames" choice. v2 has one."""
    return {
        **params,
        "from_colnames": [],
    }


def migrate_params(params):
    if "contenttype" in params:
        params = _migrate_params_v0_to_v1(params)
    if "from_colnames" not in params:
        params = _migrate_params_v1_to_v2(params)
    return params
