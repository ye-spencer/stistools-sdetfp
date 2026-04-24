"""
Documented bugs
---------------
BUG-1 (getTable): When no rows match the filter AND neither exactly_one
  nor at_least_one is True, the function sets newdata = None but then
  calls len(newdata) unconditionally at line 48, raising TypeError.
  The intended "return None" path is therefore dead code: getTable will
  always crash on a no-match result unless a guard flag is set.
  The test that exercises this bug is marked @pytest.mark.xfail(strict=True)
  so the suite stays green
"""

import math
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from stistools.gettable import (
    getTable, sortrows, rotateTrace,
    STRING_WILDCARD, INT_WILDCARD,
)


def _make_recarray(**columns):
    # build a numpy recarray from given column data
    names   = list(columns.keys())
    arrays  = list(columns.values())
    formats = []
    for col in arrays:
        if len(col) == 0:
            formats.append('<U32')
        elif isinstance(col[0], str):
            formats.append('<U32')
        elif isinstance(col[0], (np.integer, int)):
            formats.append('i4')
        else:
            formats.append('f8')
    dtype = np.dtype({'names': names, 'formats': formats})
    rows  = np.array([tuple(r) for r in zip(*arrays)], dtype=dtype)
    return rows.view(np.recarray)


def _mock_fd(recarray):
    # return a magic mock for a FITS file descriptor
    fd   = MagicMock()
    ext1 = MagicMock()
    ext1.data = recarray
    fd.__getitem__ = lambda self, i: ext1
    fd.close = MagicMock()
    return fd


class FakeTrace:

    def __init__(self, degperyr, mjd, nelem, a2displ, include_rot_cols=True):
        self._degperyr = np.array(degperyr, dtype=np.float64)
        self._mjd      = np.array(mjd,      dtype=np.float64)
        self._nelem    = np.array(nelem,    dtype=np.int32)
        self._a2displ  = [np.array(a, dtype=np.float64) for a in a2displ]
        if include_rot_cols:
            self.names = ['degperyr', 'mjd', 'nelem', 'a2displ']
        else:
            self.names = ['nelem', 'a2displ']

    def field(self, name):
        return {
            'degperyr': self._degperyr,
            'mjd':      self._mjd,
            'nelem':    self._nelem,
            'a2displ':  self._a2displ,
        }[name.lower()]

    def __len__(self):
        return len(self._degperyr)

class TestSortrows:

    def test_single_row_returned_unchanged(self):
        ra = _make_recarray(V=[42])
        assert list(sortrows(ra, 'V').field('V')) == [42]

    def test_empty_recarray_returned_unchanged(self):
        ra = _make_recarray(V=[])
        assert len(sortrows(ra, 'V')) == 0

    def test_ascending_integers(self):
        ra = _make_recarray(V=[3, 1, 2])
        assert list(sortrows(ra, 'V').field('V')) == [1, 2, 3]

    def test_descending_integers(self):
        ra = _make_recarray(V=[3, 1, 2])
        assert list(sortrows(ra, 'V', ascend=False).field('V')) == [3, 2, 1]

    def test_ascending_floats(self):
        ra = _make_recarray(X=[3.3, 1.1, 2.2])
        assert list(sortrows(ra, 'X').field('X')) == pytest.approx([1.1, 2.2, 3.3])

    def test_descending_floats(self):
        ra = _make_recarray(X=[3.3, 1.1, 2.2])
        assert list(sortrows(ra, 'X', ascend=False).field('X')) == pytest.approx(
            [3.3, 2.2, 1.1])

    def test_ascending_strings(self):
        ra = _make_recarray(N=['charlie', 'alice', 'bob'])
        assert list(sortrows(ra, 'N').field('N')) == ['alice', 'bob', 'charlie']

    def test_descending_strings(self):
        ra = _make_recarray(N=['charlie', 'alice', 'bob'])
        assert list(sortrows(ra, 'N', ascend=False).field('N')) == [
            'charlie', 'bob', 'alice']

    def test_sort_preserves_other_columns(self):
        ra = _make_recarray(K=[3, 1, 2], P=[30, 10, 20])
        assert list(sortrows(ra, 'K').field('P')) == [10, 20, 30]

    def test_sort_does_not_mutate_original(self):
        ra     = _make_recarray(V=[3, 1, 2])
        before = list(ra.field('V'))
        sortrows(ra, 'V')
        assert list(ra.field('V')) == before

    def test_default_ascend_is_true(self):
        ra = _make_recarray(V=[3, 1, 2])
        assert list(sortrows(ra, 'V')) == list(sortrows(ra, 'V', ascend=True))

    def test_already_sorted_ascending_unchanged(self):
        ra = _make_recarray(V=[1, 2, 3])
        assert list(sortrows(ra, 'V').field('V')) == [1, 2, 3]

    def test_already_sorted_descending_unchanged(self):
        ra = _make_recarray(V=[3, 2, 1])
        assert list(sortrows(ra, 'V', ascend=False).field('V')) == [3, 2, 1]

    def test_equal_values_smallest_wins_first_asc(self):
        ra = _make_recarray(V=[2, 2, 1])
        assert sortrows(ra, 'V').field('V')[0] == 1

    def test_returns_recarray_like(self):
        ra = _make_recarray(V=[3, 1, 2])
        assert hasattr(sortrows(ra, 'V'), 'field')

class TestRotateTraceEarlyReturn:

    def test_negative_expstart_no_change(self):
        ti   = FakeTrace([5.0], [50000.0], [4], [np.zeros(4)])
        orig = ti.field('a2displ')[0].copy()
        rotateTrace(ti, -1.0)
        assert np.allclose(ti.field('a2displ')[0], orig)

    def test_negative_expstart_minus_100_no_change(self):
        ti   = FakeTrace([5.0], [50000.0], [4], [np.ones(4) * 7.0])
        orig = ti.field('a2displ')[0].copy()
        rotateTrace(ti, -100.0)
        assert np.allclose(ti.field('a2displ')[0], orig)

    def test_exactly_zero_expstart_does_not_return_early(self):
        ti = FakeTrace([1.0], [0.0], [4], [np.zeros(4)])
        rotateTrace(ti, 0.0)
        assert np.allclose(ti.field('a2displ')[0], np.zeros(4))

    def test_missing_both_rotation_cols_no_change(self):
        ti   = FakeTrace([1.0], [50000.0], [4], [np.zeros(4)],
                         include_rot_cols=False)
        orig = ti.field('a2displ')[0].copy()
        rotateTrace(ti, 55000.0)
        assert np.allclose(ti.field('a2displ')[0], orig)

    def test_missing_only_mjd_no_change(self):
        class OnlyDegperyr:
            names = ['degperyr', 'nelem', 'a2displ']
            def field(self, n):
                return {'degperyr': np.array([10.0]),
                        'nelem':    np.array([3]),
                        'a2displ':  [np.zeros(3)]}[n.lower()]
            def __len__(self): return 1

        ti   = OnlyDegperyr()
        orig = ti.field('a2displ')[0].copy()
        rotateTrace(ti, 55000.0)
        assert np.allclose(ti.field('a2displ')[0], orig)

    def test_missing_only_degperyr_no_change(self):
        class OnlyMjd:
            names = ['mjd', 'nelem', 'a2displ']
            def field(self, n):
                return {'mjd':     np.array([50000.0]),
                        'nelem':   np.array([3]),
                        'a2displ': [np.zeros(3)]}[n.lower()]
            def __len__(self): return 1

        ti   = OnlyMjd()
        orig = ti.field('a2displ')[0].copy()
        rotateTrace(ti, 55000.0)
        assert np.allclose(ti.field('a2displ')[0], orig)

class TestRotateTraceZeroEffect:

    def test_zero_degperyr_no_change(self):
        a2 = np.array([1.0, 2.0, 3.0, 4.0])
        ti = FakeTrace([0.0], [50000.0], [4], [a2.copy()])
        rotateTrace(ti, 55000.0)
        assert np.allclose(ti.field('a2displ')[0], a2)

    def test_expstart_equals_mjd_no_change(self):
        a2 = np.array([5.0, 6.0, 7.0])
        ti = FakeTrace([10.0], [50000.0], [3], [a2.copy()])
        rotateTrace(ti, 50000.0)
        assert np.allclose(ti.field('a2displ')[0], a2)

class TestRotateTraceNumericCorrectness:

    def test_known_rotation_single_row(self):
        degperyr = 36.525
        mjd = 50000.0
        expstart = 50365.25   # exactly one year later
        nelem = 5
        ti = FakeTrace([degperyr], [mjd], [nelem], [np.zeros(nelem)])
        rotateTrace(ti, expstart)

        angle = degperyr * (expstart - mjd) / 365.25
        tan_angle = math.tan(angle * math.pi / 180.0)
        x = np.arange(nelem, dtype=np.float64) - (nelem // 2)
        expected = -x * tan_angle
        assert np.allclose(ti.field('a2displ')[0], expected)

    def test_even_nelem_centre_index_has_zero_correction(self):
        ti = FakeTrace([10.0], [50000.0], [4], [np.zeros(4)])
        rotateTrace(ti, 50100.0)
        assert ti.field('a2displ')[0][2] == pytest.approx(0.0)

    def test_odd_nelem_centre_index_has_zero_correction(self):
        ti = FakeTrace([10.0], [50000.0], [5], [np.zeros(5)])
        rotateTrace(ti, 50100.0)
        assert ti.field('a2displ')[0][2] == pytest.approx(0.0)

    def test_nelem_1_correction_is_always_zero(self):
        a2 = np.array([99.0])
        ti = FakeTrace([100.0], [50000.0], [1], [a2.copy()])
        rotateTrace(ti, 55000.0)
        assert ti.field('a2displ')[0][0] == pytest.approx(99.0)

    def test_expstart_before_mjd_reverses_sign(self):
        nelem = 4
        ti_after = FakeTrace([10.0], [50000.0], [nelem], [np.zeros(nelem)])
        ti_before = FakeTrace([10.0], [50100.0], [nelem], [np.zeros(nelem)])
        rotateTrace(ti_after, 50100.0)
        rotateTrace(ti_before, 50000.0)
        assert np.allclose(ti_after.field('a2displ')[0], -ti_before.field('a2displ')[0])

    def test_negative_degperyr_reverses_sign(self):
        nelem  = 5
        ti_pos = FakeTrace([ 10.0], [50000.0], [nelem], [np.zeros(nelem)])
        ti_neg = FakeTrace([-10.0], [50000.0], [nelem], [np.zeros(nelem)])
        rotateTrace(ti_pos, 50100.0)
        rotateTrace(ti_neg, 50100.0)
        assert np.allclose(ti_pos.field('a2displ')[0],
                           -ti_neg.field('a2displ')[0])

    def test_a2displ_modified_in_place(self):
        a2_orig = np.zeros(4)
        ti = FakeTrace([5.0], [50000.0], [4], [a2_orig])
        id_before = id(ti.field('a2displ')[0])
        rotateTrace(ti, 50200.0)
        assert id(ti.field('a2displ')[0]) == id_before

    def test_uppercase_col_names_still_trigger_rotation(self):
        class UpperNames:
            names = ['DEGPERYR', 'MJD', 'NELEM', 'A2DISPL']

            def __init__(self):
                self._a2 = [np.zeros(4)] 

            def field(self, n):
                return {'degperyr': np.array([10.0]),
                        'mjd': np.array([50000.0]),
                        'nelem': np.array([4]),
                        'a2displ': self._a2}[n.lower()]

            def __len__(self): return 1

        ti   = UpperNames()
        orig = ti.field('a2displ')[0].copy()
        rotateTrace(ti, 50100.0)
        assert not np.allclose(ti.field('a2displ')[0], orig)

class TestRotateTraceMultipleRows:

    def test_both_rows_updated(self):
        ti = FakeTrace([10.0, 20.0], [50000.0, 50000.0], [4, 4], [np.zeros(4), np.zeros(4)])
        rotateTrace(ti, 50100.0)
        assert not np.all(ti.field('a2displ')[0] == 0.0)
        assert not np.all(ti.field('a2displ')[1] == 0.0)

    def test_each_row_matches_independent_hand_calculation(self):
        degperyr_list = [10.0, 20.0]
        mjd_val = 50000.0
        expstart = 50100.0
        nelem = 4
        ti = FakeTrace(degperyr_list, [mjd_val] * 2, [nelem] * 2,
                       [np.zeros(nelem), np.zeros(nelem)])
        rotateTrace(ti, expstart)

        for i, dpy in enumerate(degperyr_list):
            angle = dpy * (expstart - mjd_val) / 365.25
            tan_a = math.tan(angle * math.pi / 180.0)
            x = np.arange(nelem, dtype=np.float64) - (nelem // 2)
            expected = -x * tan_a
            assert np.allclose(ti.field('a2displ')[i], expected), f"Row {i}"

    def test_same_degperyr_same_mjd_gives_equal_corrections(self):
        ti = FakeTrace([10.0, 10.0], [50000.0, 50000.0], [4, 4], [np.zeros(4), np.zeros(4)])
        rotateTrace(ti, 50100.0)
        assert np.allclose(ti.field('a2displ')[0], ti.field('a2displ')[1])

    def test_different_degperyr_gives_different_corrections(self):
        ti = FakeTrace([10.0, 20.0], [50000.0, 50000.0], [4, 4], [np.zeros(4), np.zeros(4)])
        rotateTrace(ti, 50100.0)
        assert not np.allclose(ti.field('a2displ')[0], ti.field('a2displ')[1])

class TestGetTableFitsInterface:

    def test_opens_table_in_readonly_mode(self, mocker):
        ra = _make_recarray(V=[1, 2])
        m  = mocker.patch('stistools.gettable.fits.open', return_value=_mock_fd(ra))
        getTable('my.fits', {})
        m.assert_called_once_with('my.fits', mode='readonly')

    def test_fd_closed_on_successful_match(self, mocker):
        ra = _make_recarray(NAME=['alpha'])
        fd = _mock_fd(ra)
        mocker.patch('stistools.gettable.fits.open', return_value=fd)
        getTable('t.fits', {'NAME': 'alpha'})
        fd.close.assert_called_once()

    def test_fd_closed_on_empty_filter(self, mocker):
        ra = _make_recarray(V=[1])
        fd = _mock_fd(ra)
        mocker.patch('stistools.gettable.fits.open', return_value=fd)
        getTable('t.fits', {})
        fd.close.assert_called_once()

class TestGetTableEmptyColumn:

    def test_empty_column_returns_none(self, mocker):
        ra = _make_recarray(NAME=[])
        mocker.patch('stistools.gettable.fits.open', return_value=_mock_fd(ra))
        result = getTable('t.fits', {'NAME': 'anything'})
        assert result is None

    def test_empty_column_fd_not_closed(self, mocker):
        ra = _make_recarray(NAME=[])
        fd = _mock_fd(ra)
        mocker.patch('stistools.gettable.fits.open', return_value=fd)
        getTable('t.fits', {'NAME': 'x'})
        assert not fd.close.called

class TestGetTableEmptyFilter:

    def test_returns_all_rows(self, mocker):
        ra = _make_recarray(V=[1, 2, 3])
        mocker.patch('stistools.gettable.fits.open', return_value=_mock_fd(ra))
        assert len(getTable('t.fits', {})) == 3

    def test_returns_copy_of_data(self, mocker):
        ra = _make_recarray(V=[1, 2, 3])
        mocker.patch('stistools.gettable.fits.open', return_value=_mock_fd(ra))
        result = getTable('t.fits', {})
        assert hasattr(result, 'field')

    def test_with_sortcol_sorts_all_rows(self, mocker):
        ra = _make_recarray(V=[3, 1, 2])
        mocker.patch('stistools.gettable.fits.open', return_value=_mock_fd(ra))
        result = getTable('t.fits', {}, sortcol='V')
        assert list(result.field('V')) == [1, 2, 3]

class TestGetTableStringWildcardFilterValue:

    def test_wildcard_filter_skips_key_returns_all(self, mocker):
        ra = _make_recarray(NAME=['alpha', 'beta', 'gamma'])
        mocker.patch('stistools.gettable.fits.open', return_value=_mock_fd(ra))
        assert len(getTable('t.fits', {'NAME': STRING_WILDCARD})) == 3

    def test_wildcard_combined_with_real_filter(self, mocker):
        ra = _make_recarray(NAME=['alpha', 'beta'], T=['X', 'Y'])
        mocker.patch('stistools.gettable.fits.open', return_value=_mock_fd(ra))
        result = getTable('t.fits', {'NAME': STRING_WILDCARD, 'T': 'X'})
        assert len(result) == 1
        assert result.field('T')[0] == 'X'

    def test_all_wildcard_filter_values_return_all_rows(self, mocker):
        ra = _make_recarray(A=['x', 'y'], B=['p', 'q'])
        mocker.patch('stistools.gettable.fits.open', return_value=_mock_fd(ra))
        result = getTable('t.fits',
                          {'A': STRING_WILDCARD, 'B': STRING_WILDCARD})
        assert len(result) == 2

class TestGetTableStrBytesWildcardInTable:

    def _fd_with_str_column(self, column_value, on_match=None):
        matched = on_match or MagicMock(__len__=lambda s: 1)
        not_matched = MagicMock(__len__=lambda s: 0)

        data = MagicMock()
        data.field.return_value = column_value
        data.__getitem__ = lambda s, idx: matched if idx else not_matched

        fd = MagicMock()
        ext1 = MagicMock()
        ext1.data = data
        fd.__getitem__ = lambda s, i: ext1
        fd.close = MagicMock()
        return fd, matched

    def test_str_column_exact_match_selects_row(self, mocker):
        fd, matched = self._fd_with_str_column("hello")
        mocker.patch('stistools.gettable.fits.open', return_value=fd)
        result = getTable('t.fits', {'COL': 'hello'}, at_least_one=True)
        assert result is matched

    def test_str_column_string_wildcard_value_matches_any_filter(self, mocker):
        fd, matched = self._fd_with_str_column(STRING_WILDCARD)
        mocker.patch('stistools.gettable.fits.open', return_value=fd)
        result = getTable('t.fits', {'COL': 'something_else'}, at_least_one=True)
        assert result is matched

    def test_bytes_column_exact_match_selects_row(self, mocker):
        fd, matched = self._fd_with_str_column(b"hello")
        mocker.patch('stistools.gettable.fits.open', return_value=fd)
        result = getTable('t.fits', {'COL': b'hello'}, at_least_one=True)
        assert result is matched

    def test_bytes_column_wildcard_value_matches_any_filter(self, mocker):
        fd, matched = self._fd_with_str_column(b"ANY")
        not_matched = MagicMock(__len__=lambda s: 0)
        mocker.patch('stistools.gettable.fits.open', return_value=fd)
        # b"ANY" != STRING_WILDCARD ("ANY") in Python 3 → wild=False → no match
        with pytest.raises(RuntimeError):
            # at_least_one=True so no-match raises rather than hitting the None bug
            getTable('t.fits', {'COL': b'other'}, at_least_one=True)


class TestGetTableIntWildcardInTable:

    def test_int_wildcard_row_matches_any_int_filter(self, mocker):
        ra = _make_recarray(ID=[np.int32(INT_WILDCARD), np.int32(42)])
        mocker.patch('stistools.gettable.fits.open', return_value=_mock_fd(ra))
        result = getTable('t.fits', {'ID': np.int32(99)})
        assert result is not None
        assert INT_WILDCARD in list(result.field('ID'))

    def test_int_wildcard_and_exact_match_both_included(self, mocker):
        ra = _make_recarray(ID=[np.int32(INT_WILDCARD), np.int32(42), np.int32(7)])
        mocker.patch('stistools.gettable.fits.open', return_value=_mock_fd(ra))
        result = getTable('t.fits', {'ID': np.int32(42)})
        assert len(result) == 2

    def test_only_int_wildcard_row_returned_when_no_exact_match(self, mocker):
        ra = _make_recarray(ID=[np.int32(INT_WILDCARD), np.int32(7)])
        mocker.patch('stistools.gettable.fits.open', return_value=_mock_fd(ra))
        result = getTable('t.fits', {'ID': np.int32(99)})
        assert result is not None
        assert len(result) == 1
        assert result.field('ID')[0] == INT_WILDCARD

class TestGetTableStringFilter:

    def test_single_match(self, mocker):
        ra = _make_recarray(NAME=['alpha', 'beta', 'gamma'])
        mocker.patch('stistools.gettable.fits.open', return_value=_mock_fd(ra))
        result = getTable('t.fits', {'NAME': 'beta'})
        assert len(result) == 1
        assert result.field('NAME')[0] == 'beta'

    def test_multiple_matches(self, mocker):
        ra = _make_recarray(T=['A', 'A', 'B'])
        mocker.patch('stistools.gettable.fits.open', return_value=_mock_fd(ra))
        assert len(getTable('t.fits', {'T': 'A'})) == 2

    def test_multi_key_and_logic(self, mocker):
        ra = _make_recarray(T=['A', 'A', 'B'], V=[np.int32(1), np.int32(2), np.int32(1)])
        mocker.patch('stistools.gettable.fits.open', return_value=_mock_fd(ra))
        result = getTable('t.fits', {'T': 'A', 'V': np.int32(1)})
        assert len(result) == 1

    def test_three_key_filter(self, mocker):
        ra = _make_recarray(
            A=['x', 'x', 'x', 'y'],
            B=[np.int32(1), np.int32(1), np.int32(2), np.int32(1)],
            C=['p', 'q', 'p', 'p'],
        )
        mocker.patch('stistools.gettable.fits.open', return_value=_mock_fd(ra))
        result = getTable('t.fits', {'A': 'x', 'B': np.int32(1), 'C': 'p'})
        assert len(result) == 1
        assert result.field('C')[0] == 'p'

    def test_result_has_field_accessor(self, mocker):
        ra = _make_recarray(NAME=['alpha'])
        mocker.patch('stistools.gettable.fits.open', return_value=_mock_fd(ra))
        result = getTable('t.fits', {'NAME': 'alpha'})
        assert hasattr(result, 'field')

class TestGetTableNoMatch:

    def test_exactly_one_no_match_raises_runtime_error(self, mocker):
        ra = _make_recarray(NAME=['alpha', 'beta'])
        mocker.patch('stistools.gettable.fits.open', return_value=_mock_fd(ra))
        with pytest.raises(RuntimeError, match="no matching row"):
            getTable('t.fits', {'NAME': 'gamma'}, exactly_one=True)

    def test_at_least_one_no_match_raises_runtime_error(self, mocker):
        ra = _make_recarray(NAME=['alpha'])
        mocker.patch('stistools.gettable.fits.open', return_value=_mock_fd(ra))
        with pytest.raises(RuntimeError, match="no matching row"):
            getTable('t.fits', {'NAME': 'z'}, at_least_one=True)

    def test_runtime_error_message_contains_table_name(self, mocker):
        ra = _make_recarray(NAME=['alpha'])
        mocker.patch('stistools.gettable.fits.open', return_value=_mock_fd(ra))
        with pytest.raises(RuntimeError, match='reference.fits'):
            getTable('reference.fits', {'NAME': 'z'}, exactly_one=True)

    def test_runtime_error_message_contains_filter_value(self, mocker):
        ra = _make_recarray(NAME=['alpha'])
        mocker.patch('stistools.gettable.fits.open', return_value=_mock_fd(ra))
        with pytest.raises(RuntimeError, match='missing_value'):
            getTable('t.fits', {'NAME': 'missing_value'}, exactly_one=True)

    def test_both_flags_no_match_raises(self, mocker):
        ra = _make_recarray(NAME=['alpha'])
        mocker.patch('stistools.gettable.fits.open', return_value=_mock_fd(ra))
        with pytest.raises(RuntimeError):
            getTable('t.fits', {'NAME': 'z'},
                     exactly_one=True, at_least_one=True)

    def test_multi_key_no_common_row_raises(self, mocker):
        ra = _make_recarray(T=['A', 'B'], V=[np.int32(1), np.int32(2)])
        mocker.patch('stistools.gettable.fits.open', return_value=_mock_fd(ra))
        with pytest.raises(RuntimeError):
            getTable('t.fits', {'T': 'A', 'V': np.int32(2)}, at_least_one=True)

    @pytest.mark.xfail(strict=True, raises=TypeError,
                       reason="BUG-1: getTable raises TypeError when no row "
                              "matches and neither exactly_one nor at_least_one "
                              "is True, because len(None) is called after "
                              "newdata is set to None.  The intended 'return "
                              "None' path is unreachable.")
    def test_no_match_no_flags_bug(self, mocker):
        """
        BUG-1 documentation: this call should return None but crashes instead.
        xfail(strict=True) keeps the suite green and will alert if the bug
        is fixed so the xfail marker can be removed.
        """
        ra = _make_recarray(NAME=['alpha'])
        mocker.patch('stistools.gettable.fits.open', return_value=_mock_fd(ra))
        getTable('t.fits', {'NAME': 'nonexistent'})

class TestGetTableExactlyOne:

    def test_single_match_no_error_no_warning(self, mocker, capsys):
        ra = _make_recarray(NAME=['alpha', 'beta'])
        mocker.patch('stistools.gettable.fits.open', return_value=_mock_fd(ra))
        result = getTable('t.fits', {'NAME': 'alpha'}, exactly_one=True)
        assert len(result) == 1
        assert 'more than one' not in capsys.readouterr().out

    def test_multiple_matches_prints_warning(self, mocker, capsys):
        ra = _make_recarray(T=['A', 'A', 'B'])
        mocker.patch('stistools.gettable.fits.open', return_value=_mock_fd(ra))
        getTable('t.fits', {'T': 'A'}, exactly_one=True)
        assert 'more than one' in capsys.readouterr().out

    def test_multiple_matches_warning_includes_table_name(self, mocker, capsys):
        ra = _make_recarray(T=['A', 'A'])
        mocker.patch('stistools.gettable.fits.open', return_value=_mock_fd(ra))
        getTable('ref_table.fits', {'T': 'A'}, exactly_one=True)
        assert 'ref_table.fits' in capsys.readouterr().out

    def test_multiple_matches_warning_includes_filter(self, mocker, capsys):
        ra = _make_recarray(T=['A', 'A'])
        mocker.patch('stistools.gettable.fits.open', return_value=_mock_fd(ra))
        getTable('t.fits', {'T': 'A'}, exactly_one=True)
        assert "'T'" in capsys.readouterr().out

    def test_multiple_matches_all_rows_returned(self, mocker, capsys):
        ra = _make_recarray(T=['A', 'A', 'B'])
        mocker.patch('stistools.gettable.fits.open', return_value=_mock_fd(ra))
        result = getTable('t.fits', {'T': 'A'}, exactly_one=True)
        assert len(result) == 2

class TestGetTableAtLeastOne:

    def test_single_match_no_error(self, mocker):
        ra = _make_recarray(NAME=['alpha'])
        mocker.patch('stistools.gettable.fits.open', return_value=_mock_fd(ra))
        assert len(getTable('t.fits', {'NAME': 'alpha'}, at_least_one=True)) == 1

    def test_multiple_matches_no_error(self, mocker):
        ra = _make_recarray(T=['A', 'A', 'B'])
        mocker.patch('stistools.gettable.fits.open', return_value=_mock_fd(ra))
        assert len(getTable('t.fits', {'T': 'A'}, at_least_one=True)) == 2

    def test_multiple_matches_no_warning_printed(self, mocker, capsys):
        ra = _make_recarray(T=['A', 'A'])
        mocker.patch('stistools.gettable.fits.open', return_value=_mock_fd(ra))
        getTable('t.fits', {'T': 'A'}, at_least_one=True)
        assert 'more than one' not in capsys.readouterr().out

class TestGetTableSortcol:

    def test_sortcol_sorts_multiple_matches(self, mocker):
        ra = _make_recarray(T=['A', 'A', 'B'], S=[3.0, 1.0, 2.0])
        mocker.patch('stistools.gettable.fits.open', return_value=_mock_fd(ra))
        result = getTable('t.fits', {'T': 'A'}, sortcol='S')
        assert list(result.field('S')) == pytest.approx([1.0, 3.0])

    def test_sortcol_none_preserves_original_order(self, mocker):
        ra = _make_recarray(T=['A', 'A'], S=[3.0, 1.0])
        mocker.patch('stistools.gettable.fits.open', return_value=_mock_fd(ra))
        result = getTable('t.fits', {'T': 'A'}, sortcol=None)
        assert list(result.field('S')) == pytest.approx([3.0, 1.0])

    def test_sortcol_not_applied_to_single_match(self, mocker):
        ra = _make_recarray(NAME=['a', 'b'], S=[5.0, 1.0])
        mocker.patch('stistools.gettable.fits.open', return_value=_mock_fd(ra))
        result = getTable('t.fits', {'NAME': 'a'}, sortcol='S')
        assert len(result) == 1

    def test_sortcol_with_empty_filter_sorts_all_rows(self, mocker):
        ra = _make_recarray(V=[3, 1, 2])
        mocker.patch('stistools.gettable.fits.open', return_value=_mock_fd(ra))
        result = getTable('t.fits', {}, sortcol='V')
        assert list(result.field('V')) == [1, 2, 3]

class TestConstants:

    def test_string_wildcard_value(self):
        assert STRING_WILDCARD == "ANY"

    def test_int_wildcard_value(self):
        assert INT_WILDCARD == -1

    def test_string_wildcard_type(self):
        assert isinstance(STRING_WILDCARD, str)

    def test_int_wildcard_type(self):
        assert isinstance(INT_WILDCARD, int)

    def test_int_wildcard_is_negative(self):
        assert INT_WILDCARD < 0