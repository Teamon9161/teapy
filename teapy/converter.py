from __future__ import annotations

import numpy as np
import pandas as pd


class Converter:
    """Convert the input type to ndarray and convert the output array
    to the input type in order to suppport more types of input"""

    def __init__(self, io_dict=None) -> None:
        self.otype = None  # type name of output
        self.arr = None
        self.name = None
        self.columns = None
        self.io_dict = {
            list: "np.ndarray",
            tuple: "np.ndarray",
            pd.Series: "pd.Series",
            pd.DataFrame: "pd.DataFrame",
            np.ndarray: "np.ndarray",
        }
        if io_dict is not None:
            assert isinstance(io_dict, dict), "io_dict must be dict"  # pragma: no cover
            self.io_dict.update(io_dict)

    def __call__(self, arr, step="in"):
        if step == "in":
            return self.process_input(arr)
        elif step == "out":
            return self.process_output(arr)
        else:
            raise ValueError("step must be in or out")  # pragma: no cover

    def process_input(self, arr) -> np.ndarray:
        _type = None
        if isinstance(arr, (list, tuple)):
            res = np.asanyarray(arr)
            _type = type(arr)
        elif isinstance(arr, pd.Series):
            self.index, self.name = arr.index, arr.name
            res = arr.values
            _type = pd.Series
        elif isinstance(arr, pd.DataFrame):
            self.index, self.columns = arr.index, arr.columns
            res = arr.values
            _type = pd.DataFrame
        elif isinstance(arr, np.ndarray):
            res = arr
            _type = np.ndarray
        else:
            raise TypeError(f"Unsupported arr type, {type(arr)}")  # pragma: no cover
        self.otype = self.io_dict[_type]
        return res

    def process_output(self, arr: np.ndarray | None):
        if arr is None:  # inplace function
            return
        if arr.ndim == 0:
            return arr.item()
        if self.otype == "np.ndarray":
            return arr
        elif self.otype == "pd.Series":
            assert arr.ndim == 1
            if arr.size == self.index.size:
                return pd.Series(arr, index=self.index, name=self.name)
            else:
                return pd.Series(arr, name=self.name)
        elif self.otype == "pd.DataFrame":
            assert arr.ndim == 2
            if arr.shape[0] == self.index.size and arr.shape[1] == self.columns.size:
                return pd.DataFrame(arr, index=self.index, columns=self.columns)
            elif arr.shape[0] == self.index.size and arr.shape[1] != self.columns.size:
                return pd.DataFrame(arr, index=self.index)
            elif arr.shape[0] != self.index.size and arr.shape[1] == self.columns.size:
                return pd.DataFrame(arr, columns=self.columns)
            else:
                return pd.DataFrame(arr)
        elif self.otype == "list":
            return arr.tolist()
        elif self.otype == "tuple":
            return tuple(arr.tolist())
        else:
            raise ValueError(
                f"Unsupported output type, {self.otype}"
            )  # pragma: no cover
