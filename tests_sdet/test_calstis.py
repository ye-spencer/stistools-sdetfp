import pytest
from stistools import calstis

def test_printoptions():
    calstis.prtOptions()
    assert True

def test_get_docs():
    calstis.getHelpAsString()
    assert True

def test_main_no_args():
    with pytest.raises(SystemExit):
        calstis.main([])