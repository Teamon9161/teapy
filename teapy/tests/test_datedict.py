from teapy import DataDict, Expr
from teapy.testing import assert_allclose, assert_allclose3
import numpy as np

def test_init():
    dd = DataDict()
    dd = DataDict({'a': [2], 'b': [45]})
    assert dd.columns == ['a', 'b']
    dd = DataDict({'a': [2], 'b': [45]}, columns=['c', 'd'])
    assert dd.columns == ['c', 'd']  # override
    dd = DataDict([[5], [7]])
    assert dd.columns == ['0', '1']
    dd = DataDict([[5], [7]], columns=['0', '1'], a=[34, 5], b=[3])
    assert dd.columns == ['0', '1', 'a', 'b']
    dd = DataDict({'a': [2], 'b': [45]}, c=[34, 5])
    assert dd.columns == ['a', 'b', 'c']
    ea, eb = Expr([1]).alias('a'), Expr([2]).alias('b')
    assert DataDict([ea, eb]).columns == ['a', 'b']
    assert DataDict([ea, eb], columns=['c', 'd']).columns == ['c', 'd']
    

def test_get_and_set_item():
    dd = DataDict()
    a = np.random.randn(100)
    dd['a'] = a
    assert_allclose3(a, dd['a'].view, dd[0].view)
    b = np.random.randn(100)
    dd['b'] = b
    assert dd[['a', 'b']].columns == dd[[0, 1]].columns == ['a', 'b']

    dd[0] = b
    assert_allclose(dd[0].view, b)
    # Expr in column 0 will be renamed
    assert dd['a'].name == 'a'
    dd[['cdsf', 'adf']] = [[4], [2]]
    assert dd['^a.*$'].columns == ['a', 'adf']
    assert dd[['b', '^a.*$']].columns == ['b', 'a', 'adf']
    
    dd['^a.*$'] = dd['^a.*$'].apply(lambda e: e*2)
    dd.eval()
    assert dd['adf'].view == 4
    assert_allclose(dd['a'].view, 2*b)
    dd[['^a.*$', 'cdsf']] = dd[['^a.*$', 'cdsf']].apply(lambda e: e/2)
    assert dd['cdsf'].eview() == 2
    
    
def test_drop():
    dd = DataDict([np.random.randn(10), np.random.randn(10)], columns=['a', 'b'])
    assert dd.drop('a').columns == ['b']
    assert dd.drop(['a', 'b']).columns == []
    dd.drop('b', inplace=True)
    assert dd.columns == ['a']
    

def test_dtypes():
    dd = DataDict(
        a=np.random.randint(1, 3, 3), 
        b=[1.0, 2.0, 3.0], 
        c=['df', '134', '231']
    )
    assert dd.dtypes == {'a': 'Int32', 'b': 'Float64', 'c': 'String'}
