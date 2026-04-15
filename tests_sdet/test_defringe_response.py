from stistools.defringe._response import get_dispaxis, response
import io
import sys
import numpy as np

# test sample = * (default) does not print anything and returns expected shape with threshold = None (default)
def test_response_sample_valid_threshold_none():
    calibration_array = np.tile(np.linspace(1, 10, 30), (5,1)).astype(np.float64)
    normalization_array = np.ones((5,30), dtype=np.float64)
    response_result = response(
        calibration_array=calibration_array, 
        normalization_array=normalization_array
    )

    assert response_result.shape == calibration_array.shape

# test sample = * does not print anything and returns expected shape with a threshold value
def test_response_sample_valid_threshold_value():
    calibration_array = np.tile(np.linspace(1, 10, 30), (5,1)).astype(np.float64)
    normalization_array = np.ones((5,30), dtype=np.float64)
    result = response(
        calibration_array=calibration_array,
        normalization_array=normalization_array,
        threshold=5
    )

    assert result.shape == calibration_array.shape

# test sample != * prints expected string
def test_response_sample_invalid():
    calibration_array = np.tile(np.linspace(1, 10, 30), (5,1)).astype(np.float64)
    normalization_array = np.ones((5,30), dtype=np.float64)
    captured_output = io.StringIO()  
    sys.stdout = captured_output 
    response(
        calibration_array=calibration_array, 
        normalization_array=normalization_array, 
        sample="invalid_sample"
    )
    sys.stdout = sys.__stdout__ 
    output = captured_output.getvalue() 
    
    assert 'Only sample="*" currently supported for this version of response' in output


# test that get_dispaxis returns 1
def test_get_dispaxis():
    assert get_dispaxis(None) == 1
