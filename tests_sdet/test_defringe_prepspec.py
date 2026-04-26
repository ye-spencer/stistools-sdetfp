import os
import pytest
from unittest.mock import MagicMock, patch, mock_open

from stistools.defringe.prepspec import prepspec, call_prepspec

# MOCKING HELPERS

def mock_getval_side_effect(filename, keyword=None, ext=0, *args, **kwargs):
    """
    A smart side_effect for fits.getval that returns the correct mock header values based on which keyword the script asks for
    """

    key = keyword if keyword else args[0] if args else None
    
    mock_headers = {
        'OPT_ELEM': 'G750L',
        'DETECTOR': 'CCD',
        'INSTRUME': 'STIS'
    }
    return mock_headers.get(key, 'UNKNOWN')


def create_mock_fits_context():
    """
    Creates a mock that behaves like a FITS file opened with a context manager
    """
    mock_hdu0 = MagicMock()
    mock_hdu0.header = {
        'STATFLAG': False,
        'DQICORR': 'OMIT', 'BLEVCORR': 'OMIT', 'BIASCORR': 'OMIT', 
        'DARKCORR': 'OMIT', 'FLATCORR': 'OMIT', 'CRCORR': 'OMIT',
        'HELCORR': 'OMIT', 'X2DCORR': 'OMIT', 'WAVECORR': 'PERFORM',
        'DARKFILE': 'dummy_dark.fits', 'PFLTFILE': 'dummy_pflt.fits',
        'CRREJTAB': 'dummy_crrejtab.fits'
    }
    
    mock_hdulist = MagicMock()
    mock_hdulist.__getitem__.return_value = mock_hdu0
    mock_hdulist.__enter__.return_value = mock_hdulist 
    mock_hdulist.__exit__.return_value = None
    
    return mock_hdulist

# BLACK-BOX TESTS

@patch('stistools.defringe.prepspec.fits.getval')
class TestPrepspecValidation:
    
    def test_invalid_detector_raises_error(self, mock_getval):
        mock_getval.side_effect = lambda f, k=None, ext=0, *a, **kw: 'WFC3' if k=='DETECTOR' else 'STIS'
        with pytest.raises(ValueError, match="Intended for use on STIS/CCD"):
            prepspec('dummy.fits')

    def test_invalid_initguess_raises_error(self, mock_getval):
        mock_getval.side_effect = mock_getval_side_effect
        with pytest.raises(ValueError, match="initguess must be in"):
            prepspec('dummy.fits', initguess='invalid_guess')

    @patch('stistools.defringe.prepspec.os.access', return_value=False)
    def test_unwritable_outroot_raises_error(self, mock_os_access, mock_getval):
        mock_getval.side_effect = mock_getval_side_effect
        with pytest.raises(IOError, match="Cannot write to"):
            prepspec('dummy.fits', outroot='/restricted/path/')


# WHITE-BOX TESTS

@patch.dict(os.environ, {'oref': '/mock/oref/'}) 
@patch('stistools.defringe.prepspec.fits.getval', side_effect=mock_getval_side_effect)
@patch('stistools.defringe.prepspec.os.access', return_value=True) 
@patch('stistools.defringe.prepspec.os.chdir')                     
@patch('stistools.defringe.prepspec.calstis', return_value=0)      
@patch('builtins.open', new_callable=mock_open, read_data="Mock log line\n") 
class TestPrepspecLogic:

    @patch('stistools.defringe.prepspec.fits.open')
    def test_g750l_header_updates(self, mock_fits_open, mock_open_log, mock_calstis, mock_chdir, mock_access, mock_getval):
        mock_fits_file = create_mock_fits_context()
        mock_fits_open.return_value = mock_fits_file
        
        prepspec('dummy.fits')

        header = mock_fits_file[0].header
        assert header['STATFLAG'] is True
        assert header['CRCORR'] == 'PERFORM'
        assert header['DQICORR'] == 'PERFORM'

    @patch('stistools.defringe.prepspec.fits.open')
    def test_g750m_header_updates(self, mock_fits_open, mock_open_log, mock_calstis, mock_chdir, mock_access, mock_getval):
        
        def g750m_side_effect(filename, keyword=None, *args, **kwargs):
            if keyword == 'OPT_ELEM': return 'G750M'
            if keyword == 'DETECTOR': return 'CCD'
            if keyword == 'INSTRUME': return 'STIS'
            return 'UNKNOWN'
            
        mock_getval.side_effect = g750m_side_effect
        
        mock_fits_file = create_mock_fits_context()
        mock_fits_open.return_value = mock_fits_file

        prepspec('dummy.fits')

        header = mock_fits_file[0].header
        assert header['HELCORR'] == 'PERFORM'
        assert header['X2DCORR'] == 'PERFORM'
        assert header['WAVECORR'] == 'OMIT'

# WHITE-BOX + MOCKING

@patch.dict(os.environ, {'oref': '/mock/oref/'})
@patch('stistools.defringe.prepspec.fits.getval', side_effect=mock_getval_side_effect)
@patch('stistools.defringe.prepspec.fits.open')
@patch('stistools.defringe.prepspec.os.access', return_value=True)
@patch('stistools.defringe.prepspec.os.chdir')
@patch('stistools.defringe.prepspec.calstis', return_value=0)
@patch('builtins.open', new_callable=mock_open, read_data="")
class TestPrepspecFileSystemIsolation:

    @patch('stistools.defringe.prepspec.shutil.rmtree') 
    @patch('stistools.defringe.prepspec.shutil.copy')
    @patch('stistools.defringe.prepspec.mkdtemp', return_value='/tmp/mockdir')
    @patch('stistools.defringe.prepspec.os.chmod')
    def test_initguess_modifies_crrejtab(self, mock_chmod, mock_mkdtemp, mock_copy, mock_rmtree, mock_open_log, mock_calstis, mock_chdir, mock_access, mock_fits_open, mock_getval):
        
        mock_fits_open.return_value = create_mock_fits_context()

        prepspec('dummy.fits', initguess='median')

        mock_mkdtemp.assert_called_once()
        mock_copy.assert_called_once()
        mock_rmtree.assert_called_once_with('/tmp/mockdir') 
        
    @patch('stistools.defringe.prepspec.shutil.rmtree') 
    @patch('stistools.defringe.prepspec.shutil.copy')
    @patch('stistools.defringe.prepspec.mkdtemp', return_value='/tmp/mockdir')
    @patch('stistools.defringe.prepspec.os.chmod')
    def test_initguess_fallback_to_oref(self, mock_chmod, mock_mkdtemp, mock_copy, mock_rmtree, mock_open_log, mock_calstis, mock_chdir, mock_access, mock_fits_open, mock_getval):
        
        mock_fits_open.return_value = create_mock_fits_context()
        mock_copy.side_effect = [FileNotFoundError, None]

        with pytest.warns(UserWarning, match="Using CRREJTAB in \\$oref instead"):
            prepspec('dummy.fits', initguess='minimum')

        assert mock_copy.call_count == 2
        fallback_path = mock_copy.call_args_list[1][0][0]
        assert '/mock/oref/' in fallback_path

class TestCallPrepspecCLI:
    
    @patch('stistools.defringe.prepspec.prepspec') 
    @patch('sys.argv', ['stistools.defringe.prepspec.py', 'input.fits', 'outdir/', '--initguess', 'median']) 
    def test_command_line_parsing(self, mock_main_func):

        call_prepspec()
        
        mock_main_func.assert_called_once()
        
        kwargs = mock_main_func.call_args[1]
        
        assert kwargs['inspec'] == 'input.fits'
        assert kwargs['outroot'] == 'outdir/'
        assert kwargs['initguess'] == 'median'
        
    @patch('stistools.defringe.prepspec.prepspec') 
    @patch('sys.argv', ['stistools.defringe.prepspec.py', 'input.fits', 'outdir/', '--initguess', 'None'])
    def test_cli_string_none_conversion(self, mock_main_func):
        call_prepspec()
        kwargs = mock_main_func.call_args[1]
        
        assert kwargs['initguess'] is None