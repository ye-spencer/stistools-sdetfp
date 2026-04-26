import pytest
from stistools import x2d

# Statement coverage: exercises prtOptions code path.
def test_printoptions():
    x2d.prtOptions()

# Statement coverage: exercises getHelpAsString code path.
def test_get_docs():
    x2d.getHelpAsString()

# Branch coverage: empty args triggers early-exit branch.
def test_main_no_args():
    with pytest.raises(SystemExit):
        x2d.main([])

# Branch coverage: invalid option triggers exception-handler exit.
def test_main_invalid_option():
    with pytest.raises(SystemExit):
        x2d.main(["--bogus"])

# Branch coverage: --version branch calls cs7.e and returns 0.
def test_main_version(mocker):
    mock_call = mocker.patch("stistools.x2d.subprocess.call", return_value=0)
    assert x2d.main(["--version"]) == 0
    mock_call.assert_called_once_with(["cs7.e", "--version"])

# Branch coverage: -r branch calls cs7.e and returns 0.
def test_main_r(mocker):
    mock_call = mocker.patch("stistools.x2d.subprocess.call", return_value=0)
    assert x2d.main(["-r"]) == 0
    mock_call.assert_called_once_with(["cs7.e", "-r"])

# Branch coverage: flags without positional args trigger nargs<1 exit.
def test_main_flags_no_input():
    with pytest.raises(SystemExit):
        x2d.main(["-t"])

# Statement/branch coverage: -t with input invokes x2d().
def test_main_with_input(mocker, tmp_path):
    mock_x2d = mocker.patch("stistools.x2d.x2d", return_value=0)
    infile = tmp_path / "in.fits"
    infile.touch()
    with pytest.raises(SystemExit):
        x2d.main(["-t", str(infile)])
    mock_x2d.assert_called_once_with(
        str(infile), output="", verbose=False, timestamps=True
    )

# Branch coverage: input+output positional args path.
def test_main_with_input_and_output(mocker, tmp_path):
    mock_x2d = mocker.patch("stistools.x2d.x2d", return_value=0)
    infile = tmp_path / "in.fits"
    infile.touch()
    with pytest.raises(SystemExit):
        x2d.main([str(infile), "out.fits"])
    mock_x2d.assert_called_once_with(
        str(infile), output="out.fits", verbose=False, timestamps=False
    )

# Branch coverage: too many positional args triggers nargs>2 exit.
def test_main_too_many_args():
    with pytest.raises(SystemExit):
        x2d.main(["a", "b", "c"])

# Branch coverage: print_version returns 0 via cs7.e.
def test_x2d_print_version(mocker):
    mock_call = mocker.patch("stistools.x2d.subprocess.call", return_value=0)
    assert x2d.x2d("in.fits", print_version=True) == 0
    mock_call.assert_called_once_with(["cs7.e", "--version"])

# Branch coverage: print_revision returns 0 via cs7.e.
def test_x2d_print_revision(mocker):
    mock_call = mocker.patch("stistools.x2d.subprocess.call", return_value=0)
    assert x2d.x2d("in.fits", print_revision=True) == 0
    mock_call.assert_called_once_with(["cs7.e", "-r"])

# Branch coverage: no glob match returns 2.
def test_x2d_no_file_match():
    assert x2d.x2d("nonexistent_xyz.fits") == 2

# Branch coverage: mismatched input/output counts returns 2.
def test_x2d_input_output_mismatch(mocker, tmp_path):
    mocker.patch("stistools.x2d.subprocess.call", return_value=0)
    infile = tmp_path / "in.fits"
    infile.touch()
    assert x2d.x2d(str(infile), output="a.fits b.fits") == 2

# Branch coverage: defaults exercise helcorr+fluxcorr perform path, status=0.
def test_x2d_defaults(mocker, tmp_path):
    mocker.patch("stistools.x2d.subprocess.call", return_value=0)
    infile = tmp_path / "in.fits"
    infile.touch()
    assert x2d.x2d(str(infile)) == 0

# Branch coverage: helcorr/fluxcorr omit triggers -x2d switch branch.
def test_x2d_no_switch(mocker, tmp_path):
    mock_call = mocker.patch("stistools.x2d.subprocess.call", return_value=0)
    infile = tmp_path / "in.fits"
    infile.touch()
    assert x2d.x2d(str(infile), helcorr="omit", fluxcorr="omit") == 0
    assert "-x2d" in mock_call.call_args[0][0]

# Branch coverage: err_alg='wgt_err' appends -wgt_err.
def test_x2d_wgt_err(mocker, tmp_path):
    mock_call = mocker.patch("stistools.x2d.subprocess.call", return_value=0)
    infile = tmp_path / "in.fits"
    infile.touch()
    x2d.x2d(str(infile), err_alg="wgt_err")
    assert "-wgt_err" in mock_call.call_args[0][0]

# Branch coverage: invalid err_alg raises RuntimeError.
def test_x2d_bad_err_alg(mocker, tmp_path):
    mocker.patch("stistools.x2d.subprocess.call", return_value=0)
    infile = tmp_path / "in.fits"
    infile.touch()
    with pytest.raises(RuntimeError):
        x2d.x2d(str(infile), err_alg="bogus")

# Branch coverage: all options + existing trailer + nonzero status returns 1.
def test_x2d_all_options(mocker, tmp_path):
    mocker.patch("stistools.x2d.subprocess.call", return_value=1)
    infile = tmp_path / "in.fits"
    infile.touch()
    trailer = tmp_path / "x2d.trl"
    trailer.touch()
    out = tmp_path / "out.fits"
    result = x2d.x2d(
        str(infile), output=str(out), blazeshift=1.5,
        verbose=True, timestamps=True, trailer=str(trailer),
    )
    assert result == 1

# Blackbox: cs7.e nonzero status surfaces as nonzero return.
def test_x2d_cs7_nonzero(mocker, tmp_path):
    mocker.patch("stistools.x2d.subprocess.call", return_value=1)
    infile = tmp_path / "in.fits"
    infile.touch()
    assert x2d.x2d(str(infile)) != 0

# Statement coverage: run() forwards a configobj to x2d().
def test_run_forwards_configobj(mocker):
    mock_x2d = mocker.patch("stistools.x2d.x2d", return_value=0)
    cfg = {
        "input": "in.fits", "output": "", "helcorr": "perform",
        "fluxcorr": "perform", "statflag": True, "center": False,
        "blazeshift": None, "err_alg": "wgt_var", "verbose": False,
        "timestamps": False, "trailer": "", "print_version": False,
        "print_revision": False,
    }
    x2d.run(cfg)
    mock_x2d.assert_called_once()
