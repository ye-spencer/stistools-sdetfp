import pytest
from stistools import calstis

def test_printoptions():
    calstis.prtOptions()
    assert True

# def test_get_docs():
#     calstis.getHelpAsString()
#     assert True

# def test_main_no_args():
#     with pytest.raises(SystemExit):
#         calstis.main([])

# def test_main_invalid_option():
#     with pytest.raises(SystemExit):
#         calstis.main(["--invalid"])

