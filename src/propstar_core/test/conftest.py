from __future__ import annotations
import pytest
import numpy as np
from astropy.io import fits
from astropy import units as u


@pytest.fixture
def sample_image() -> fits.PrimaryHDU:
    def make_sample_image() -> fits.PrimaryHDU:
        # if is_2d:
        #     data = np.ones((501, 501))
        # else:
        data = np.ones((200, 251, 251))
        ra0, dec0 = 3.794, 32.865
        hdu = fits.PrimaryHDU(data=data)
        hdu.header["CRVAL1"] = ra0
        hdu.header["CRVAL2"] = dec0
        hdu.header["CRVAL3"] = (10.0e3, "m/s")
        hdu.header["CRPIX1"] = 125
        hdu.header["CRPIX2"] = 125
        hdu.header["CRPIX3"] = 100
        hdu.header["CDELT1"] = (u.arcsec.to(u.deg), 'deg')
        hdu.header["CDELT2"] = u.arcsec.to(u.deg)
        hdu.header["CDELT3"] = 100.0
        hdu.header["CUNIT1"] = "deg"
        hdu.header["CUNIT2"] = "deg"
        hdu.header["CUNIT3"] = "m/s"
        hdu.header["CTYPE1"] = "RA---TAN"
        hdu.header["CTYPE2"] = "DEC--TAN"
        hdu.header["CTYPE3"] = "VRAD"
        hdu.header["EQUINOX"] = 2000.0
        hdu.header["RADESYS"] = ("FK5", 'Coordinate system')
        hdu.header["SPECSYS"] = ("LSRK", 'Velocity system')
        hdu.header["RESTFREQ"] = (72.78382e9, "Hz")
        hdu.header["BUNIT"] = ("Jy/Beam", "Brightness unit")
        hdu.header["BMAJ"] = 5*u.arcsec.to(u.deg)
        hdu.header["BMIN"] = 4*u.arcsec.to(u.deg)
        hdu.header["BPA"] = 22.0
        return hdu
    return make_sample_image
