from __future__ import annotations
import numpy as np
import os

from astropy.io import fits
from astropy import units as u
from radio_beam import Beam
from spectral_cube import SpectralCube as sc
import propstar_core.data_handle as data_handle
import pytest


def test_match_cube(tmp_path, sample_image) -> None:
    """
    test match_cube function. Initial test should be a cube of constant flux
    equals to 1 K,
    which after matching should still be a cube of constant flux 1 K.

    Test that using a main beam efficiency of 1.0 does not change the cube.
    Test that using a main beam efficiency of 0.1 it changes the cube by the
    expected factor of 10x.
    """
    dir = tmp_path / 'fits'
    dir.mkdir()
    file_link = os.path.join(os.fspath(dir), "test_image.fits")
    file_link2 = os.path.join(os.fspath(dir), "test_image2.fits")
    hdu = sample_image()
    beam = Beam.from_fits_header(hdu.header)
    j_to_k = beam.jtok(hdu.header["RESTFREQ"] * u.Hz)
    # normalize the cube to 1 K
    hdu.data = np.ones(hdu.data.shape) / j_to_k.value
    hdu.header["BUNIT"] = "Jy/beam"
    hdu.writeto(file_link, overwrite=True)
    new_hd = hdu.header.copy()
    new_hd["BMAJ"] = 1.5 * hdu.header["BMAJ"]
    new_hd["BMIN"] = new_hd["BMAJ"]
    new_hd["BPA"] = 0.0
    
    cube_in = sc.read(file_link)
    # capture wrong eta_mb
    with pytest.raises(ValueError):
        cube_out = data_handle.match_cube(cube_in, new_hd, eta_mb=-1.0, convert_K=True)
    # test convovlve
    cube_out = data_handle.match_cube(cube_in, new_hd, eta_mb=1.0, convert_K=True)
    np.testing.assert_allclose(cube_out.unmasked_data[:,5:-5,5:-5].value, cube_in.unmasked_data[:,5:-5,5:-5].value*j_to_k.value, rtol=0.05)
    # convovlve and right use of eta_mb
    cube_out = data_handle.match_cube(cube_in, new_hd, eta_mb=0.1, convert_K=True)
    np.testing.assert_allclose(cube_out.unmasked_data[:,5:-5,5:-5].value, 10*cube_in.unmasked_data[:,5:-5,5:-5].value*j_to_k.value, rtol=0.05)
