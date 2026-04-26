from stistools.ocrreject import ocrreject, prtOptions, getHelpAsString
import pytest

def test_print_version(monkeypatch):
    called = {}

    def fake_call(args, **kwargs):
        called["args"] = args
        return 0

    monkeypatch.setattr("subprocess.call", fake_call)

    status = ocrreject("input.fits", "out.fits", print_version=True)

    assert status == 0
    assert called["args"] == ["cs2.e", "--version"]

def test_print_revision(monkeypatch):
    called = {}

    def fake_call(args, **kwargs):
        called["args"] = args
        return 0

    monkeypatch.setattr("subprocess.call", fake_call)

    status = ocrreject("input.fits", "out.fits", print_revision=True)

    assert called["args"] == ["cs2.e", "-r"]

def test_no_matching_input(monkeypatch, capsys):

    monkeypatch.setattr("glob.glob", lambda x: [])

    status = ocrreject("bad*.fits", "out.fits")

    captured = capsys.readouterr()

    assert status == 2
    assert "No file name matched" in captured.out

def test_all_mode_multiple_outputs(monkeypatch, capsys):

    monkeypatch.setattr("glob.glob", lambda x: ["a.fits", "b.fits"])

    status = ocrreject("*.fits", "a.fits b.fits")

    captured = capsys.readouterr()

    assert status == 2
    assert "output must be exactly one file name" in captured.out


def test_all_false_mismatch(monkeypatch, capsys):

    monkeypatch.setattr("glob.glob", lambda x: ["a.fits", "b.fits"])

    status = ocrreject("*.fits", "out.fits", all=False)

    captured = capsys.readouterr()

    assert status == 2

def test_optional_args(monkeypatch):

    monkeypatch.setattr("glob.glob", lambda x: ["a.fits"])

    captured = {}

    def fake_call(args, **kwargs):
        captured["args"] = args
        return 0

    monkeypatch.setattr("subprocess.call", fake_call)

    ocrreject(
        "a.fits",
        "out.fits",
        crsigmas="5",
        crradius=1.5,
        crthresh=0.8,
        verbose=True
    )

    args = captured["args"]

    assert "-sigmas" in args
    assert "-radius" in args
    assert "-thresh" in args
    assert "-v" in args

def test_crmask_invalid(monkeypatch):

    monkeypatch.setattr("glob.glob", lambda x: ["a.fits"])

    with pytest.raises(RuntimeError):
        ocrreject("a.fits", "out.fits", crmask="maybe")

def test_all_false_processing(monkeypatch):

    monkeypatch.setattr("glob.glob", lambda x: ["a.fits", "b.fits"])

    calls = []

    def fake_call(args, **kwargs):
        calls.append(args)
        return 0

    monkeypatch.setattr("subprocess.call", fake_call)

    status = ocrreject(
        "*.fits",
        "out1.fits out2.fits",
        all=False
    )

    assert status == 0
    assert len(calls) == 2

def test_verbose_warning(monkeypatch, capsys):

    monkeypatch.setattr("glob.glob", lambda x: ["a.fits"])

    def fake_call(args, **kwargs):
        return 1

    monkeypatch.setattr("subprocess.call", fake_call)

    ocrreject("a.fits", "out.fits", verbose=True)

    captured = capsys.readouterr()

    assert "Warning" in captured.out


def test_main_not_enough_args(capsys):

    from stistools import ocrreject

    with pytest.raises(SystemExit):
        ocrreject.main([])

    out = capsys.readouterr().out
    assert "At least input and output file names must be specified." in out

def test_main_version(monkeypatch):

    called = {}

    def fake_call(args):
        called["args"] = args
        return 0

    monkeypatch.setattr("subprocess.call", fake_call)

    from stistools import ocrreject

    result = ocrreject.main(["--version", "a", "b"])

    assert called["args"] == ["cs2.e", "--version"]

def test_main_revision(monkeypatch):

    called = {}

    def fake_call(args):
        called["args"] = args
        return 0

    monkeypatch.setattr("subprocess.call", fake_call)

    from stistools import ocrreject

    ocrreject.main(["-r", "a", "b"])

    assert called["args"] == ["cs2.e", "-r"]

def test_main_runs_ocrreject(monkeypatch):


    monkeypatch.setattr("sys.exit", lambda *a, **k: None)

    called = {}

    def fake_ocrreject(*args, **kwargs):
        called["args"] = args
        return 0

    monkeypatch.setattr("stistools.ocrreject.ocrreject", fake_ocrreject)

    from stistools import ocrreject

    ocrreject.main(["input.fits", "output.fits"])

    assert called["args"][0] == "input.fits"

def test_prtOptions(capsys):

    prtOptions()

    captured = capsys.readouterr()

    assert "The command-line options are:" in captured.out

def test_crmask_yes(monkeypatch):

    monkeypatch.setattr("glob.glob", lambda x: ["a.fits"])

    captured = {}

    def fake_call(args, **kwargs):
        captured["args"] = args
        return 0

    monkeypatch.setattr("subprocess.call", fake_call)

    ocrreject("a.fits", "out.fits", crmask="yes")

    assert "-crmask" in captured["args"]

def test_trailer_file(monkeypatch, tmp_path):

    monkeypatch.setattr("glob.glob", lambda x: ["a.fits"])

    monkeypatch.setattr("subprocess.call", lambda *a, **k: 0)

    trailer = tmp_path / "test.trl"

    status = ocrreject("a.fits", "out.fits", trailer=str(trailer))

    assert status == 0

def test_badinpdq(monkeypatch):

    monkeypatch.setattr("glob.glob", lambda x: ["a.fits"])

    called = {}

    def fake_call(args, **kwargs):
        called["args"] = args
        return 0

    monkeypatch.setattr("subprocess.call", fake_call)

    ocrreject("a.fits", "out.fits", badinpdq=4)

    assert "-pdq" in called["args"]

def test_timestamps(monkeypatch):

    monkeypatch.setattr("glob.glob", lambda x: ["a.fits"])

    captured = {}

    def fake_call(args, **kwargs):
        captured["args"] = args
        return 0

    monkeypatch.setattr("subprocess.call", fake_call)

    ocrreject("a.fits", "out.fits", timestamps=True)

    assert "-t" in captured["args"]

def test_get_help():

    helptext = getHelpAsString()

    assert "cosmic rays" in helptext