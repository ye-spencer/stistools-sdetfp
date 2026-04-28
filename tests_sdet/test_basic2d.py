import pytest
from stistools import basic2d


# Statement coverage: prtOptions prints
def test_prt_options():
    basic2d.prtOptions()


# Statement coverage: getHelpAsString returns doc
def test_get_help():
    result = basic2d.getHelpAsString()
    assert result is None or isinstance(result, str)


# Branch coverage: empty args early exit
def test_main_no_args():
    with pytest.raises(SystemExit):
        basic2d.main([])


# Branch coverage: invalid option exits
def test_main_invalid_option():
    with pytest.raises(SystemExit):
        basic2d.main(["--invalid"])


# Branch coverage: --version calls cs1.e
def test_main_version(mocker):
    mock = mocker.patch("stistools.basic2d.subprocess.call", return_value=0)
    assert basic2d.main(["--version"]) == 0
    mock.assert_called_once_with(["cs1.e", "--version"])


# Branch coverage: -r calls cs1.e
def test_main_r(mocker):
    mock = mocker.patch("stistools.basic2d.subprocess.call", return_value=0)
    assert basic2d.main(["-r"]) == 0
    mock.assert_called_once_with(["cs1.e", "-r"])


# Branch coverage: flags without positional exits
def test_main_flags_no_file():
    with pytest.raises(SystemExit):
        basic2d.main(["-v", "x", "-t"])


# Branch coverage: too many positional args exits
def test_main_too_many_args():
    with pytest.raises(SystemExit):
        basic2d.main(["a", "b", "c"])


# Statement/branch coverage: main dispatches to basic2d
def test_main_with_file(mocker, tmp_path):
    mock_b2d = mocker.patch("stistools.basic2d.basic2d", return_value=0)
    infile = tmp_path / "raw.fits"
    infile.touch()
    with pytest.raises(SystemExit):
        basic2d.main(["-v", "x", "-t", str(infile)])
    mock_b2d.assert_called_once_with(
        str(infile), output="", outblev="",
        verbose=True, timestamps=True)


# Branch coverage: two positional args sets output
def test_main_with_output(mocker, tmp_path):
    mock_b2d = mocker.patch("stistools.basic2d.basic2d", return_value=0)
    infile = tmp_path / "raw.fits"
    infile.touch()
    with pytest.raises(SystemExit):
        basic2d.main([str(infile), "out.fits"])
    assert mock_b2d.call_args.kwargs["output"] == "out.fits"


# Branch coverage: print_version calls cs1.e --version
def test_basic2d_print_version(mocker):
    mock = mocker.patch("stistools.basic2d.subprocess.call", return_value=0)
    assert basic2d.basic2d("f.fits", print_version=True) == 0
    mock.assert_called_once_with(["cs1.e", "--version"])


# Branch coverage: print_revision calls cs1.e -r
def test_basic2d_print_revision(mocker):
    mock = mocker.patch("stistools.basic2d.subprocess.call", return_value=0)
    assert basic2d.basic2d("f.fits", print_revision=True) == 0
    mock.assert_called_once_with(["cs1.e", "-r"])


# Branch coverage: no matching glob returns 2
def test_basic2d_no_match():
    assert basic2d.basic2d("nonexistent_xyz_zzz.fits") == 2


# Branch coverage: valid file defaults returns 0
def test_basic2d_default(mocker, tmp_path):
    mocker.patch("stistools.basic2d.subprocess.call", return_value=0)
    infile = tmp_path / "raw.fits"
    infile.touch()
    assert basic2d.basic2d(str(infile)) == 0


# Branch coverage: output count mismatch returns 2
def test_basic2d_output_mismatch(tmp_path):
    infile = tmp_path / "raw.fits"
    infile.touch()
    assert basic2d.basic2d(str(infile), output="a.fits b.fits") == 2


# Branch coverage: outblev count mismatch returns 2
def test_basic2d_outblev_mismatch(tmp_path):
    infile = tmp_path / "raw.fits"
    infile.touch()
    assert basic2d.basic2d(str(infile), outblev="a.txt b.txt") == 2


# Branch coverage: all perform flags appended
def test_basic2d_all_perform(mocker, tmp_path):
    mock = mocker.patch("stistools.basic2d.subprocess.call", return_value=0)
    infile = tmp_path / "raw.fits"
    infile.touch()
    basic2d.basic2d(str(infile), verbose=True, timestamps=True, darkscale="1.0")
    argv = mock.call_args.args[0]
    for flag in ["-v", "-t", "-dscl", "-dqi", "-blev", "-dopp", "-lors",
                 "-glin", "-lflg", "-bias", "-dark", "-flat", "-phot", "-stat"]:
        assert flag in argv


# Branch coverage: all omit flags absent from argv
def test_basic2d_all_omit(mocker, tmp_path):
    mock = mocker.patch("stistools.basic2d.subprocess.call", return_value=0)
    infile = tmp_path / "raw.fits"
    infile.touch()
    basic2d.basic2d(
        str(infile),
        dqicorr="omit", blevcorr="omit", doppcorr="omit",
        lorscorr="omit", glincorr="omit", lflgcorr="omit",
        biascorr="omit", darkcorr="omit", flatcorr="omit",
        photcorr="omit", statflag=False)
    argv = mock.call_args.args[0]
    for flag in ["-dqi", "-blev", "-dopp", "-lors", "-glin", "-lflg",
                 "-bias", "-dark", "-flat", "-phot", "-stat", "-v", "-t", "-dscl"]:
        assert flag not in argv


# Branch coverage: output arg appended
def test_basic2d_output_appended(mocker, tmp_path):
    mock = mocker.patch("stistools.basic2d.subprocess.call", return_value=0)
    infile = tmp_path / "raw.fits"
    infile.touch()
    basic2d.basic2d(str(infile), output="out.fits")
    assert "out.fits" in mock.call_args.args[0]


# Branch coverage: outblev arg appended
def test_basic2d_outblev_appended(mocker, tmp_path):
    mock = mocker.patch("stistools.basic2d.subprocess.call", return_value=0)
    infile = tmp_path / "raw.fits"
    infile.touch()
    basic2d.basic2d(str(infile), outblev="blev.txt")
    assert "blev.txt" in mock.call_args.args[0]


# Branch coverage: trailer opens new file
def test_basic2d_trailer_new(mocker, tmp_path):
    mocker.patch("stistools.basic2d.subprocess.call", return_value=0)
    infile = tmp_path / "raw.fits"
    infile.touch()
    trailer = tmp_path / "new.trl"
    basic2d.basic2d(str(infile), trailer=str(trailer))
    assert trailer.exists()


# Branch coverage: existing trailer + verbose prints append msg
def test_basic2d_trailer_existing_verbose(mocker, tmp_path, capsys):
    mocker.patch("stistools.basic2d.subprocess.call", return_value=0)
    infile = tmp_path / "raw.fits"
    infile.touch()
    trailer = tmp_path / "exists.trl"
    trailer.touch()
    basic2d.basic2d(str(infile), trailer=str(trailer), verbose=True)
    assert "Appending" in capsys.readouterr().out


# Branch coverage: non-zero status sets cumulative_status
def test_basic2d_nonzero_status(mocker, tmp_path):
    mocker.patch("stistools.basic2d.subprocess.call", return_value=1)
    infile = tmp_path / "raw.fits"
    infile.touch()
    assert basic2d.basic2d(str(infile)) == 1


# Branch coverage: verbose + non-zero status prints warning
def test_basic2d_verbose_warning(mocker, tmp_path, capsys):
    mocker.patch("stistools.basic2d.subprocess.call", return_value=1)
    infile = tmp_path / "raw.fits"
    infile.touch()
    basic2d.basic2d(str(infile), verbose=True)
    assert "Warning" in capsys.readouterr().out


# Branch coverage: verbose prints running line
def test_basic2d_verbose_running(mocker, tmp_path, capsys):
    mocker.patch("stistools.basic2d.subprocess.call", return_value=0)
    infile = tmp_path / "raw.fits"
    infile.touch()
    basic2d.basic2d(str(infile), verbose=True)
    assert "Running basic2d" in capsys.readouterr().out


# Statement coverage: run delegates to basic2d
def test_run_delegates(mocker):
    mock_b2d = mocker.patch("stistools.basic2d.basic2d")
    cfg = {k: "" for k in [
        "input", "output", "outblev", "dqicorr", "atodcorr", "blevcorr",
        "doppcorr", "lorscorr", "glincorr", "lflgcorr", "biascorr",
        "darkcorr", "flatcorr", "shadcorr", "photcorr", "darkscale", "trailer"]}
    cfg.update({"statflag": True, "verbose": False, "timestamps": False,
                "print_version": False, "print_revision": False})
    basic2d.run(cfg)
    mock_b2d.assert_called_once()
