# Teapy
[![Build](https://github.com/teamon9161/teapy/workflows/Build/badge.svg)](https://github.com/teamon9161/teapy/actions)
[![PyPI](https://img.shields.io/pypi/v/teapy)](https://pypi.org/project/teapy)
[![codecov](https://codecov.io/gh/teamon9161/teapy/branch/master/graph/badge.svg?token=WK0F7P1VC6)](https://codecov.io/gh/teamon9161/teapy)

## Blazingly fast datadict library in Python

Teapy is a high-performance data dictionary library implemented in Rust, designed for blazingly fast operations. It offers the following features:
* Lazy evaluation
* Handling of NaN values
* Multi-threaded processing
* Support for any dimensionality

## Setup
Install the latest teapy version with:
`pip install teapy`


## Basic Usage

### Creating Expressions
```Python
# Expressions can be created in various ways
import numpy as np
import pandas as pd
import polars as pl
import teapy as tp

e1 = tp.Expr([1, 2, 3])  # Create from list
e2 = tp.Expr((1, 2, 3))  # Create from tuple
e3 = tp.Expr(np.array([1, 2, 3]), 'e3')  # Create from numpy.ndarray, name is e3
e4 = tp.Expr(pd.Series([1, 2, 3]))  # Create from pandas.Series
e5 = tp.Expr(pl.Series([1, 2, 3]))  # Create from polars.Series
```
### Creating DataDicts
```Python
# DataDicts can be created in different ways
dd1 = tp.DataDict({'a': [1, 2], 'b': [2, 3]}, c=[3, 4])  # Create from dictionary
dd2 = tp.DataDict([tp.Expr([1, 2], 'a'), tp.Expr([2, 3], 'b')])  # Create from list of expressions
dd3 = tp.DataDict(a=[1, 2], b=[2, 3], c=np.array([3, 6, 2]))  # Create by specifying key-value pairs
```

### Evaluating Expressions and DataDicts
```Python
# Evaluating Expressions
e = tp.Expr([1, 2, 3]).mean()
e.eval()  # Execute the expression
e.view  # View the memory of the array
e.eview()  # Execute the expression and view the memory of the array
e.value()  # Execute the expression and copy the memory of the array to a new numpy.ndarray

# Evaluating DataDicts
dd = tp.DataDict({'a': [1, 2]*10, 'b': [2, 3]*10}, c=[3, 4])
dd = dd.select([
    dd['a'].ts_mean(3).alias('d'), 
    dd['b'].ts_std(4).alias('e')
])
dd.eval(['d', 'e'])  # Evaluate specific keys in parallel
dd.eval()  # Or evaluate all expressions in parallel
print(dd['d'])

```