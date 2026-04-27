import numpy as np
import pytest
from astropy.io import fits
from pathlib import Path

from stistools.ocrreject_exam import ocrreject_exam, BoxExtended

def make_flt_file(path, shape=(20, 10)):

    primary = fits.PrimaryHDU()
    hdr = primary.header

    hdr["INSTRUME"] = "STIS"
    hdr["DETECTOR"] = "CCD"
    hdr["OBSMODE"] = "ACCUM"
    hdr["PROPOSID"] = 12345
    hdr["ROOTNAME"] = "testobs"

    hdr["CRSPLIT"] = 1
    hdr["NRPTEXP"] = 2
    hdr["EXPTIME"] = 100.0
    hdr["NEXTEND"] = 6

    data1 = np.ones(shape) * 100
    data2 = np.ones(shape) * 105

    sci1 = fits.ImageHDU(data1, name="SCI")
    sci1.header["EXPTIME"] = 100.0

    sci2 = fits.ImageHDU(data2, name="SCI")
    sci2.header["EXPTIME"] = 100.0

    err1 = fits.ImageHDU(np.ones(shape), name="ERR")
    dq1  = fits.ImageHDU(np.zeros(shape, dtype=int), name="DQ")

    err2 = fits.ImageHDU(np.ones(shape), name="ERR")
    dq2  = fits.ImageHDU(np.zeros(shape, dtype=int), name="DQ")

    fits.HDUList([
        primary,
        sci1, err1, dq1,
        sci2, err2, dq2
    ]).writeto(path)

def make_flt_file_with_crs(path, shape=(20, 10),
                           cr_inside=None, cr_outside=None):
    """Create a minimal FLT file with optional CR flags."""
    if cr_inside is None:
        cr_inside = [(5, 2)]
    if cr_outside is None:
        cr_outside = [(10, 3)]

    primary = fits.PrimaryHDU()
    hdr = primary.header
    hdr["INSTRUME"] = "STIS"
    hdr["DETECTOR"] = "CCD"
    hdr["OBSMODE"] = "ACCUM"
    hdr["PROPOSID"] = 12345
    hdr["ROOTNAME"] = "testobs"
    hdr["CRSPLIT"] = 1
    hdr["NRPTEXP"] = 2
    hdr["EXPTIME"] = 100.0
    hdr["NEXTEND"] = 6

    data = np.ones(shape) * 100
    sci1 = fits.ImageHDU(data.copy(), name="SCI")
    sci1.header["EXPTIME"] = 100.0
    sci2 = fits.ImageHDU(data.copy(), name="SCI")
    sci2.header["EXPTIME"] = 100.0

    err1 = fits.ImageHDU(np.ones(shape), name="ERR")
    err2 = fits.ImageHDU(np.ones(shape), name="ERR")

    dq1 = np.zeros(shape, dtype=int)
    dq2 = np.zeros(shape, dtype=int)
    cr_bit = 8192  # 2**13

    for r, c in cr_inside:
        dq1[r, c] |= cr_bit
        dq2[r, c] |= cr_bit
    for r, c in cr_outside:
        dq1[r, c] |= cr_bit
        dq2[r, c] |= cr_bit

    dq1_hdu = fits.ImageHDU(dq1, name="DQ")
    dq2_hdu = fits.ImageHDU(dq2, name="DQ")

    fits.HDUList([
        primary,
        sci1, err1, dq1_hdu,
        sci2, err2, dq2_hdu,
    ]).writeto(path)

def make_sx1_file(path, width=10):

    extrlocy = np.ones(width) * 5
    extrsize = np.ones(width) * 4

    cols = [
        fits.Column(name="EXTRLOCY", format=f"{width}E", array=[extrlocy]),
        fits.Column(name="EXTRSIZE", format=f"{width}E", array=[extrsize]),
    ]

    table = fits.BinTableHDU.from_columns(cols)

    fits.HDUList([fits.PrimaryHDU(), table]).writeto(path)


def test_missing_sx1_file(tmp_path):

    obs = "testobs"
    flt = tmp_path / f"{obs}_flt.fits"

    make_flt_file(flt)

    with pytest.raises(FileNotFoundError):
        ocrreject_exam(obs, data_dir=str(tmp_path))


def test_invalid_header(tmp_path):

    obs = "badobs"
    flt = tmp_path / f"{obs}_flt.fits"
    sx1 = tmp_path / f"{obs}_sx1.fits"

    primary = fits.PrimaryHDU()
    primary.header["INSTRUME"] = "NOTSTIS"

    fits.HDUList([primary]).writeto(flt)

    make_sx1_file(sx1)

    with pytest.raises(ValueError):
        ocrreject_exam(obs, data_dir=str(tmp_path))


def test_ocrreject_basic(tmp_path):
    obs = "testobs"
    flt = tmp_path / f"{obs}_flt.fits"
    sx1 = tmp_path / f"{obs}_sx1.fits"
    make_flt_file_with_crs(flt)    
    make_sx1_file(sx1)
    result = ocrreject_exam(obs, data_dir=str(tmp_path))
    assert isinstance(result, list)
    assert result[0]["rootname"] == obs
    assert result[0]["avg_ratio"] > 0

def test_plot_branch(tmp_path):
    obs = "testobs"
    flt = tmp_path / f"{obs}_flt.fits"
    sx1 = tmp_path / f"{obs}_sx1.fits"
    make_flt_file_with_crs(flt)
    make_sx1_file(sx1)
    result = ocrreject_exam(obs, data_dir=str(tmp_path), plot=True)
    assert isinstance(result, list)

def test_multiple_exposures(tmp_path):
    obs = "testobs"
    flt = tmp_path / f"{obs}_flt.fits"
    sx1 = tmp_path / f"{obs}_sx1.fits"
    make_flt_file_with_crs(flt)
    make_sx1_file(sx1)
    result = ocrreject_exam(obs, data_dir=str(tmp_path))
    assert isinstance(result, list)

def test_bad_extraction_box(tmp_path):

    obs = "testobs"

    flt = tmp_path / f"{obs}_flt.fits"
    sx1 = tmp_path / f"{obs}_sx1.fits"

    make_flt_file(flt, shape=(5, 10))

    width = 10
    extrlocy = np.ones(width) * 100
    extrsize = np.ones(width) * 50

    cols = [
        fits.Column(name="EXTRLOCY", format=f"{width}E", array=[extrlocy]),
        fits.Column(name="EXTRSIZE", format=f"{width}E", array=[extrsize]),
    ]

    table = fits.BinTableHDU.from_columns(cols)
    fits.HDUList([fits.PrimaryHDU(), table]).writeto(sx1)

    with pytest.raises(BoxExtended):
        ocrreject_exam(obs, data_dir=str(tmp_path))