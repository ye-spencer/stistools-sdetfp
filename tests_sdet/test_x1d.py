import pytest

from stistools import x1d


# Statement coverage: prtOptions code path
def test_printoptions():
    x1d.prtOptions()
    assert True


# Statement coverage: getHelpAsString code path
def test_get_docs():
    x1d.getHelpAsString()
    assert True


# Branch coverage: empty args exits
def test_main_no_args():
    with pytest.raises(SystemExit):
        x1d.main([])


# Branch coverage: invalid option exits
def test_main_invalid_option():
    with pytest.raises(SystemExit):
        x1d.main(["--bogus"])


# Branch coverage: --version flag calls cs6.e
def test_main_version(mocker):
    mock_call = mocker.patch("stistools.x1d.subprocess.call", return_value=0)

    assert x1d.main(["--version"]) == 0
    mock_call.assert_called_once_with(["cs6.e", "--version"])


# Branch coverage: -r flag calls cs6.e
def test_main_r(mocker):
    mock_call = mocker.patch("stistools.x1d.subprocess.call", return_value=0)

    assert x1d.main(["-r"]) == 0
    mock_call.assert_called_once_with(["cs6.e", "-r"])


# Branch coverage: flags without fileinput exit
def test_main_flags_no_fileinput():
    with pytest.raises(SystemExit):
        x1d.main(["-t", "-v", "1"])


# Statement/branch coverage: flags with fileinput
def test_main_flags_with_fileinput(mocker, tmp_path):
    mock_x1d = mocker.patch("stistools.x1d.x1d", return_value=0)
    infile = tmp_path / "test_flt.fits"
    infile.touch()

    with pytest.raises(SystemExit):
        x1d.main(["-t", "-v", "1", str(infile)])

    mock_x1d.assert_called_once_with(
        str(infile), output="", verbose=True, timestamps=True
    )


# Branch coverage: second positional arg is output
def test_main_with_output(mocker, tmp_path):
    mock_x1d = mocker.patch("stistools.x1d.x1d", return_value=0)
    infile = tmp_path / "test_flt.fits"
    infile.touch()

    with pytest.raises(SystemExit):
        x1d.main([str(infile), "out_x1d.fits"])

    mock_x1d.assert_called_once_with(
        str(infile), output="out_x1d.fits", verbose=False, timestamps=False
    )


# Branch coverage: too many args exits
def test_main_too_many_args():
    with pytest.raises(SystemExit):
        x1d.main(["a", "b", "c"])


# Branch coverage: print_version calls cs6.e
def test_x1d_print_version(mocker):
    mock_call = mocker.patch("stistools.x1d.subprocess.call", return_value=0)

    assert x1d.x1d("test.fits", print_version=True) == 0
    mock_call.assert_called_once_with(["cs6.e", "--version"])


# Branch coverage: print_revision calls cs6.e
def test_x1d_print_revision(mocker):
    mock_call = mocker.patch("stistools.x1d.subprocess.call", return_value=0)

    assert x1d.x1d("test.fits", print_revision=True) == 0
    mock_call.assert_called_once_with(["cs6.e", "-r"])


# Branch coverage: no glob match returns 2
def test_x1d_no_file_match():
    assert x1d.x1d("nonexistent_xyz.fits") == 2


# Branch coverage: mismatched output count returns 2
def test_x1d_mismatched_output_count(mocker, tmp_path):
    infile = tmp_path / "test_flt.fits"
    infile.touch()

    assert x1d.x1d(str(infile), output="a.fits b.fits") == 2


# Branch coverage: defaults perform all corrs
def test_x1d_defaults(mocker, tmp_path):
    mocker.patch("stistools.x1d.subprocess.call", return_value=0)
    infile = tmp_path / "test_flt.fits"
    infile.touch()

    assert x1d.x1d(str(infile)) == 0


# Branch coverage: no switch set appends -x1d
def test_x1d_no_switch_set(mocker, tmp_path):
    mock_call = mocker.patch("stistools.x1d.subprocess.call", return_value=0)
    infile = tmp_path / "test_flt.fits"
    infile.touch()

    x1d.x1d(
        str(infile),
        backcorr="omit", ctecorr="omit", dispcorr="omit",
        helcorr="omit", fluxcorr="omit",
    )
    argv = mock_call.call_args[0][0]
    assert "-x1d" in argv


# Branch coverage: all numeric options + sc2d
def test_x1d_all_numeric_options(mocker, tmp_path):
    mock_call = mocker.patch("stistools.x1d.subprocess.call", return_value=0)
    infile = tmp_path / "test_flt.fits"
    infile.touch()
    trailer = tmp_path / "test.trl"
    trailer.touch()

    result = x1d.x1d(
        str(infile), output="out.fits",
        sporder=1, a2center=512.0, maxsrch=5.0, globalx=True,
        extrsize=7.0, bk1size=3.0, bk2size=3.0,
        bk1offst=10.0, bk2offst=-10.0, bktilt=0.5,
        backord=1, bksmode="average", bksorder=2,
        blazeshift=0.1, algorithm="sc2d", xoffset=1.0,
        verbose=True, timestamps=True, trailer=str(trailer),
    )
    assert result == 0


# Branch coverage: bksmode off appends -bn
def test_x1d_bksmode_off(mocker, tmp_path):
    mock_call = mocker.patch("stistools.x1d.subprocess.call", return_value=0)
    infile = tmp_path / "test_flt.fits"
    infile.touch()

    x1d.x1d(str(infile), bksmode="off")
    argv = mock_call.call_args[0][0]
    assert "-bn" in argv


# Branch coverage: invalid bksmode raises
def test_x1d_invalid_bksmode(mocker, tmp_path):
    mocker.patch("stistools.x1d.subprocess.call", return_value=0)
    infile = tmp_path / "test_flt.fits"
    infile.touch()

    with pytest.raises(RuntimeError):
        x1d.x1d(str(infile), bksmode="bogus")


# Branch coverage: invalid algorithm raises
def test_x1d_invalid_algorithm(mocker, tmp_path):
    mocker.patch("stistools.x1d.subprocess.call", return_value=0)
    infile = tmp_path / "test_flt.fits"
    infile.touch()

    with pytest.raises(RuntimeError):
        x1d.x1d(str(infile), algorithm="bogus")


# Blackbox: cs6.e non-zero status
def test_x1d_cs6_non_zero(mocker, tmp_path):
    mocker.patch("stistools.x1d.subprocess.call", return_value=1)
    infile = tmp_path / "test_flt.fits"
    infile.touch()

    assert x1d.x1d(str(infile), verbose=True) == 1


# Branch coverage: cs6.e non-zero without verbose
def test_x1d_cs6_non_zero_no_verbose(mocker, tmp_path):
    mocker.patch("stistools.x1d.subprocess.call", return_value=1)
    infile = tmp_path / "test_flt.fits"
    infile.touch()

    assert x1d.x1d(str(infile)) == 1


# Branch coverage: empty comma-split output entries
def test_x1d_output_empty_comma_entries(mocker, tmp_path):
    mocker.patch("stistools.x1d.subprocess.call", return_value=0)
    infile = tmp_path / "test_flt.fits"
    infile.touch()

    assert x1d.x1d(str(infile), output=",out.fits,") == 0


# Branch coverage: whitespace-only output skips loop
def test_x1d_output_whitespace_only(mocker, tmp_path):
    mocker.patch("stistools.x1d.subprocess.call", return_value=0)
    infile = tmp_path / "test_flt.fits"
    infile.touch()

    assert x1d.x1d(str(infile), output="   ") == 0


# Branch coverage: trailer set without verbose
def test_x1d_trailer_no_verbose(mocker, tmp_path):
    mocker.patch("stistools.x1d.subprocess.call", return_value=0)
    infile = tmp_path / "test_flt.fits"
    infile.touch()
    trailer = tmp_path / "test.trl"
    trailer.touch()

    assert x1d.x1d(str(infile), trailer=str(trailer)) == 0


# Branch coverage: bksmode None skips block
def test_x1d_bksmode_none(mocker, tmp_path):
    mocker.patch("stistools.x1d.subprocess.call", return_value=0)
    infile = tmp_path / "test_flt.fits"
    infile.touch()

    assert x1d.x1d(str(infile), bksmode=None) == 0


# Branch coverage: algorithm None skips block
def test_x1d_algorithm_none(mocker, tmp_path):
    mocker.patch("stistools.x1d.subprocess.call", return_value=0)
    infile = tmp_path / "test_flt.fits"
    infile.touch()

    assert x1d.x1d(str(infile), algorithm=None) == 0


# Statement coverage: run TEAL interface
def test_run(mocker):
    mock_x1d = mocker.patch("stistools.x1d.x1d", return_value=0)
    configobj = {
        "input": "in.fits", "output": "", "backcorr": "perform",
        "ctecorr": "perform", "dispcorr": "perform", "helcorr": "perform",
        "fluxcorr": "perform", "sporder": None, "a2center": None,
        "maxsrch": None, "globalx": False, "extrsize": None,
        "bk1size": None, "bk2size": None, "bk1offst": None,
        "bk2offst": None, "bktilt": None, "backord": None,
        "bksmode": "median", "bksorder": 3, "blazeshift": None,
        "algorithm": "unweighted", "xoffset": None, "verbose": False,
        "timestamps": False, "trailer": "", "print_version": False,
        "print_revision": False,
    }
    x1d.run(configobj)
    mock_x1d.assert_called_once()
