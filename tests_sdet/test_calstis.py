import pytest
from stistools import calstis

# Statement coverage: exercises prtOptions code path.
def test_printoptions():
    calstis.prtOptions()
    assert True

# Statement coverage: exercises getHelpAsString code path.
def test_get_docs():
    calstis.getHelpAsString()
    assert True

# Branch coverage: empty args triggers the early-exit branch in main().
def test_main_no_args():
    with pytest.raises(SystemExit):
        calstis.main([])

# Branch coverage: invalid option triggers the exception-handler exit branch.
def test_main_invalid_option():
    with pytest.raises(SystemExit):
        calstis.main(["--invalid"])

# Branch coverage: --version branch calls cs0.e correctly and returns 0.
def test_main_version(mocker):
    mock_call = mocker.patch("stistools.calstis.subprocess.call", return_value=0)

    assert calstis.main(["--version"]) == 0
    mock_call.assert_called_once_with(["cs0.e", "--version"])

# Branch coverage: -r branch calls cs0.e correctly and returns 0.
def test_main_r(mocker):
    mock_call = mocker.patch("stistools.calstis.subprocess.call", return_value=0)

    assert calstis.main(["-r"]) == 0
    mock_call.assert_called_once_with(["cs0.e", "-r"])


# Branch coverage: all non-exit flags (-v, -t, -s, -w) with no positional arg trigger the nargs < 1 exit branch.
def test_main_all_options_no_fileinput():
    with pytest.raises(SystemExit):
        calstis.main(["-v", "-t", "-s", "-w"])

# Statement/branch coverage: all non-exit flags with a valid input file exercise the full calstis() call path.
def test_main_all_options_with_fileinput(mocker, tmp_path):
    mock_calstis = mocker.patch("stistools.calstis.calstis", return_value=0)
    infile = tmp_path / "test_raw.fits"
    infile.touch()

    with pytest.raises(SystemExit):
        calstis.main(["-v", "-t", "-s", "-w", "test.wav", str(infile)])

    mock_calstis.assert_called_once_with(
        str(infile), wavecal="test.wav", outroot="",
        savetmp=True, verbose=True, timestamps=True
    )