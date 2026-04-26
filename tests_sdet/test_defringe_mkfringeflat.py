import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from stistools.defringe.mkfringeflat import mkfringeflat, get_flat_data, call_mkfringeflat

# MOCKING HELPERS

def create_mock_fits_hdulist(opt_elem='G750M', aperture='52X0.2', is_science=True):
    """
    Creates a mock FITS HDUList object with the necessary headers and data arrays
    """
    fake_data = np.ones((100, 1200), dtype=np.float64) 
    
    prihdr = {
        'opt_elem': opt_elem,
        'binaxis1': 1,
        'binaxis2': 1,
        'centera2': 50,
        'sizaxis2': 100,
        'aperture': aperture
    }
    
    hdr = {
        'ltv1': 0.0,
        'ltv2': 0.0
    }

    hdu0 = MagicMock()
    hdu0.header = prihdr
    hdu0.data = fake_data if not is_science else None

    hdu1 = MagicMock()
    hdu1.header = hdr
    hdu1.data = fake_data

    hdulist = MagicMock()
    hdulist.__len__ = lambda self: 2
    hdulist.__getitem__ = lambda self, i: [hdu0, hdu1][i]
    
    return hdulist

# TESTS

class TestGetFlatData:
    def test_returns_shifted_flat_if_provided(self):
        """If a shifted flat array is already provided, it should just return it."""
        fake_shifted = np.array([1, 2, 3])
        result = get_flat_data('dummy.fits', fake_shifted)
        assert np.array_equal(result, fake_shifted)

    @patch('stistools.defringe.mkfringeflat.fits.open')
    def test_reads_from_file_if_shifted_flat_is_none(self, mock_fits_open):
        """If shifted_flat is None, it should open the file and return HDU 1 data."""
        mock_hdulist = create_mock_fits_hdulist()
        mock_fits_open.return_value = mock_hdulist
        
        result = get_flat_data('dummy.fits', None)
        
        mock_fits_open.assert_called_once_with('dummy.fits')
        assert result.shape == (100, 1200)

class TestMkFringeFlat:

    @patch('stistools.defringe.mkfringeflat.fits.writeto')
    @patch('stistools.defringe.mkfringeflat.fits.open')
    @patch('stistools.defringe.mkfringeflat.find_loc')
    @patch('stistools.defringe.mkfringeflat.response')
    @patch('stistools.defringe.mkfringeflat.np.std', return_value=0.5)
    def test_mkfringeflat_g750m_happy_path(self, mock_std, mock_response, mock_find_loc, mock_fits_open, mock_fits_writeto):
        sci_mock = create_mock_fits_hdulist(opt_elem='G750M')
        flt_mock = create_mock_fits_hdulist(opt_elem='G750M', is_science=False)
        
        mock_fits_open.side_effect = [sci_mock, flt_mock, flt_mock, flt_mock] 
        mock_find_loc.return_value = 50.0
        mock_response.return_value = np.ones((10, 1000)) 

        mkfringeflat(
            inspec='science.fits', inflat='flat.fits', outflat='out.fits',
            do_shift=True, beg_shift=-0.1, end_shift=0.1, shift_step=0.1, 
            do_scale=True, beg_scale=0.9, end_scale=1.1, scale_step=0.1
        )

        assert mock_fits_open.call_count >= 2
        mock_find_loc.assert_called_once()
        assert mock_fits_writeto.call_count == 2
        last_write_args = mock_fits_writeto.call_args_list[-1]
        assert last_write_args[0][0] == 'out.fits'

    @patch('stistools.defringe.mkfringeflat.fits.writeto') 
    @patch('stistools.defringe.mkfringeflat.fits.open')
    @patch('stistools.defringe.mkfringeflat.find_loc')
    @patch('stistools.defringe.mkfringeflat.response')
    @patch('stistools.defringe.mkfringeflat.np.std', return_value=0.5)
    def test_mkfringeflat_g750l_branch(self, mock_std, mock_response, mock_find_loc, mock_fits_open, mock_fits_writeto):
        mock_fits_open.side_effect = [
            create_mock_fits_hdulist(opt_elem='G750L'), 
            create_mock_fits_hdulist(opt_elem='G750L', is_science=False),
            create_mock_fits_hdulist(opt_elem='G750L', is_science=False)
        ]
        mock_find_loc.return_value = 50.0
        mock_response.return_value = np.ones((10, 1000))

        mkfringeflat('sci.fits', 'flt.fits', 'out.fits', do_scale=False, shift_step=0.5)

        mock_response.assert_called()


class TestCallMkFringeFlat:
    @patch('stistools.defringe.mkfringeflat.mkfringeflat') # Mock the main function so it doesn't actually run
    @patch('sys.argv', ['mkfringeflat.py', 'input_sci.fits', 'input_flt.fits', 'output.fits', '--skip_shift'])
    def test_command_line_parsing(self, mock_main_func):
        """Tests that argparse correctly maps command line flags to kwargs."""
        
        call_mkfringeflat()
        
        mock_main_func.assert_called_once()
        kwargs = mock_main_func.call_args[1]
        
        assert kwargs['inspec'] == 'input_sci.fits'
        assert kwargs['inflat'] == 'input_flt.fits'
        assert kwargs['outflat'] == 'output.fits'
        assert kwargs['do_shift'] is False

class TestMkFringeFlatShiftLogic:
    
    @patch('stistools.defringe.mkfringeflat.fits.writeto')
    @patch('stistools.defringe.mkfringeflat.fits.open')
    @patch('stistools.defringe.mkfringeflat.find_loc', return_value=50.0)
    @patch('stistools.defringe.mkfringeflat.response')
    @patch('stistools.defringe.mkfringeflat.np.std')
    def test_shift_best_in_middle(self, mock_std, mock_response, mock_find_loc, mock_fits_open, mock_fits_writeto):
        """
        Covers the `do_shift` block where the minimum RMS is found in the MIDDLE 
        of the shift range, triggering the weighted average 'else' logic.
        """
        mock_fits_open.side_effect = [
            create_mock_fits_hdulist('G750M', is_science=True),
            create_mock_fits_hdulist('G750M', is_science=False),
            create_mock_fits_hdulist('G750M', is_science=False)
        ]
        
        mock_response.return_value = np.ones((10, 1000))
        mock_std.side_effect = [0.8, 0.2, 0.8] 

        mkfringeflat(
            'sci.fits', 'flt.fits', 'out.fits', 
            do_shift=True, 
            beg_shift=-0.1, end_shift=0.1, shift_step=0.1, 
            do_scale=False # Skip scale block to isolate our test
        )

        assert mock_std.call_count == 3
        mock_fits_writeto.assert_called()
        assert mock_fits_writeto.call_args[0][0] == 'flt_sh.fits'

    @patch('stistools.defringe.mkfringeflat.fits.writeto')
    @patch('stistools.defringe.mkfringeflat.fits.open')
    @patch('stistools.defringe.mkfringeflat.find_loc', return_value=50.0)
    @patch('stistools.defringe.mkfringeflat.response')
    @patch('stistools.defringe.mkfringeflat.np.std') 
    def test_shift_best_on_edge(self, mock_std, mock_response, mock_find_loc, mock_fits_open, mock_fits_writeto, capsys):
        """
        Covers the `do_shift` block where the minimum RMS is found on the EDGE of the shift range, triggering the 'WARNING' print statement.
        """
        mock_fits_open.side_effect = [
            create_mock_fits_hdulist('G750M', is_science=True),
            create_mock_fits_hdulist('G750M', is_science=False),
            create_mock_fits_hdulist('G750M', is_science=False)
        ]
        
        mock_response.return_value = np.ones((10, 1000))
        mock_std.side_effect = [0.1, 0.5, 0.9] 

        mkfringeflat(
            'sci.fits', 'flt.fits', 'out.fits', 
            do_shift=True, 
            beg_shift=-0.1, end_shift=0.1, shift_step=0.1, 
            do_scale=False
        )

        captured = capsys.readouterr()
        
        assert "WARNING: Best shift found on the edge" in captured.out
    
    @patch('stistools.defringe.mkfringeflat.fits.writeto')
    @patch('stistools.defringe.mkfringeflat.fits.open')
    @patch('stistools.defringe.mkfringeflat.find_loc', return_value=50.0)
    @patch('stistools.defringe.mkfringeflat.response')
    @patch('stistools.defringe.mkfringeflat.np.std', return_value=0.5)
    def test_shift_g750l_branch(self, mock_std, mock_response, mock_find_loc, mock_fits_open, mock_fits_writeto):
        mock_fits_open.side_effect = [
            create_mock_fits_hdulist(opt_elem='G750L', is_science=True),
            create_mock_fits_hdulist(opt_elem='G750L', is_science=False),
            create_mock_fits_hdulist(opt_elem='G750L', is_science=False)
        ]
        
        mock_response.return_value = np.ones((10, 1000))

        mkfringeflat(
            'sci.fits', 'flt.fits', 'out.fits', 
            do_shift=True, 
            beg_shift=0.0, end_shift=0.0, shift_step=0.1, 
            do_scale=False 
        )

        assert mock_response.call_count == 1
        kwargs = mock_response.call_args[1]
        assert kwargs['order'] == 15

class TestMkFringeFlatEdgeCases:
    
    @patch('stistools.defringe.mkfringeflat.fits.open')
    @patch('stistools.defringe.mkfringeflat.find_loc', return_value=50.0)
    def test_shift_unsupported_opt_elem_crashes(self, mock_find_loc, mock_fits_open):
        mock_fits_open.side_effect = lambda *args, **kwargs: create_mock_fits_hdulist(opt_elem='UNKNOWN')
        
        with pytest.raises(UnboundLocalError):
            mkfringeflat('sci.fits', 'flt.fits', 'out.fits', do_shift=True, do_scale=False)

    @patch('stistools.defringe.mkfringeflat.fits.open')
    @patch('stistools.defringe.mkfringeflat.find_loc', return_value=50.0)
    def test_scale_unsupported_opt_elem_crashes(self, mock_find_loc, mock_fits_open):
        mock_fits_open.side_effect = lambda *args, **kwargs: create_mock_fits_hdulist(opt_elem='UNKNOWN')
        
        with pytest.raises(UnboundLocalError):
            mkfringeflat('sci.fits', 'flt.fits', 'out.fits', do_shift=False, do_scale=True)


class TestMkFringeFlatScaleLogic:

    @patch('stistools.defringe.mkfringeflat.fits.writeto')
    @patch('stistools.defringe.mkfringeflat.fits.open')
    @patch('stistools.defringe.mkfringeflat.find_loc', return_value=50.0)
    @patch('stistools.defringe.mkfringeflat.response')
    @patch('stistools.defringe.mkfringeflat.np.std', return_value=0.5) 
    def test_scale_g750l_branch(self, mock_std, mock_response, mock_find_loc, mock_fits_open, mock_fits_writeto):
        mock_fits_open.side_effect = lambda *args, **kwargs: create_mock_fits_hdulist(opt_elem='G750L')
        
        mock_response.return_value = np.ones((10, 1000))

        mkfringeflat(
            'sci.fits', 'flt.fits', 'out.fits', 
            do_shift=False, 
            do_scale=True, beg_scale=0.9, end_scale=1.1, scale_step=0.1 
        )

        assert mock_response.call_count == 3
        kwargs = mock_response.call_args[1]
        assert kwargs['order'] == 15
    
    @patch('stistools.defringe.mkfringeflat.fits.writeto')
    @patch('stistools.defringe.mkfringeflat.fits.open')
    @patch('stistools.defringe.mkfringeflat.find_loc', return_value=50.0)
    @patch('stistools.defringe.mkfringeflat.response')
    @patch('stistools.defringe.mkfringeflat.np.std') 
    def test_scale_weighted_average_deep_middle(self, mock_std, mock_response, mock_find_loc, mock_fits_open, mock_fits_writeto):
        mock_fits_open.side_effect = lambda *args, **kwargs: create_mock_fits_hdulist(opt_elem='G750M')
        mock_response.return_value = np.ones((10, 1000))
        mock_std.side_effect = [0.8, 0.6, 0.1, 0.5, 0.9]

        mkfringeflat(
            'sci.fits', 'flt.fits', 'out.fits', 
            do_shift=False, 
            do_scale=True, beg_scale=0.8, end_scale=1.2, scale_step=0.1 
        )

        assert mock_std.call_count == 5
        
        mock_fits_writeto.assert_called()

    @patch('stistools.defringe.mkfringeflat.fits.writeto')
    @patch('stistools.defringe.mkfringeflat.fits.open')
    @patch('stistools.defringe.mkfringeflat.find_loc', return_value=50.0)
    @patch('stistools.defringe.mkfringeflat.response')
    @patch('stistools.defringe.mkfringeflat.np.std') 
    def test_scale_weighted_average_shallow_middle(self, mock_std, mock_response, mock_find_loc, mock_fits_open, mock_fits_writeto):
        mock_fits_open.side_effect = lambda *args, **kwargs: create_mock_fits_hdulist(opt_elem='G750M')
        mock_response.return_value = np.ones((10, 1000))
        mock_std.side_effect = [0.8, 0.1, 0.6, 0.5, 0.9]

        mkfringeflat(
            'sci.fits', 'flt.fits', 'out.fits', 
            do_shift=False, 
            do_scale=True, beg_scale=0.8, end_scale=1.2, scale_step=0.1 
        )

        assert mock_std.call_count == 5