from __future__ import annotations
import astropy.units as u
import numpy as np
from radio_beam import Beam
from astropy.io import fits
from spectral_cube import SpectralCube as sc
from astropy.convolution import Gaussian1DKernel


def match_hd(
    hd_in: fits.header.Header, hd_tmp: fits.header.Header
) -> fits.header.Header:
    """
    Match the spatial information of two cubes. It smoothes the first cube to
    the beam of the second cube and reprojects it to the spatial information
    of the second cube. It also returns a cube with a limited velocity range.
    Parameters:
    hd_in: header of the first cube.
    hd_tmp: header of the second cube.
    Returns:
    hd_match: header of the matched cube.
    """
    key_list = [
        "NAXIS1",
        "NAXIS2",
        "CRPIX1",
        "CRPIX2",
        "CDELT1",
        "CDELT2",
        "CRVAL1",
        "CRVAL2",
        "CTYPE1",
        "CTYPE2",
        "CUNIT1",
        "CUNIT2",
    ]
    for key_i in key_list:
        hd_in[key_i] = hd_tmp[key_i]
    hd_in.remove("PC1_1", ignore_missing=True)
    hd_in.remove("PC2_1", ignore_missing=True)
    hd_in.remove("PC1_2", ignore_missing=True)
    hd_in.remove("PC2_2", ignore_missing=True)
    return hd_in


def match_cube(
    cube: spectral_cube.spectral_cube.SpectralCube,
    # file_out: str,
    ref_hd: fits.header.Header,
    eta_mb: float = 1.0,
    convert_K: bool = True,
) -> spectral_cube.spectral_cube.SpectralCube:
    """
    Match the spatial information of two cubes. It smoothes the first cube to
    the beam of the second cube and reprojects it to the spatial information
    of the second cube. It also returns a cube corrected by the main beam
    efficiency if requested.

    Parameters:
    cube_in: spectral_cube.spectral_cube.SpectralCube
        SpectralCube cube to be matched.
    ref_hd: fits.header.Header
        Header of the second cube to match spatial information.
    eta_mb: float, optional
        Main beam efficiency factor, default is 1.0.

    Returns:
        SpectralCube
    """
    if eta_mb < 0.0 or eta_mb > 1.0:
        raise ValueError("eta_mb should be between 0 and 1.")
    new_beam = Beam.from_fits_header(ref_hd)
    # convolve the cube to the new beam and correct by the main beam efficiency
    cube_inter = cube.convolve_to(new_beam) / eta_mb
    new_hd = match_hd(cube_inter.header, ref_hd)
    match_cube = cube_inter.reproject(new_hd)
    if convert_K:
        match_cube = match_cube.to(u.K)
    return match_cube


def regrid_cube(
    cube: spectral_cube.spectral_cube.SpectralCube,
    vel_min: u.Quantity = 0 * u.km / u.s,
    vel_max: u.Quantity = 20 * u.km / u.s,
    vel_chan: u.Quantity = 0.1 * u.km / u.s,
) -> spectral_cube.spectral_cube.SpectralCube:
    """
    Regrid the spectral axis of the cube to a new velocity axis.

    Parameters:
    cube: spectral_cube.spectral_cube.SpectralCube
        Input spectral cube to be regridded.
    vel_min: u.Quantity, optional
        Minimum velocity for the new spectral axis, default is 0 km/s.
    vel_max: u.Quantity, optional
        Maximum velocity for the new spectral axis, default is 20 km/s.
    vel_chan: u.Quantity, optional
        Channel width for the new spectral axis, default is 0.1 km/s.

    Returns:
    spectral_cube.spectral_cube.SpectralCube
        Regridded spectral cube.
    """
    # spectral axis interpolation is done in velocity space (km/s units)
    cube_inter = cube.with_spectral_unit(u.km / u.s, velocity_convention="radio", )
    old_axis = cube_inter.spectral_axis
    new_axis = (
        np.arange(
            vel_min.to(u.km / u.s).value,
            vel_max.to(u.km / u.s).value,
            vel_chan.to(u.km / u.s).value,
        )
        * u.km
        / u.s
    )
    fwhm_factor = np.sqrt(8 * np.log(2))
    channel_in = np.abs(old_axis[1] - old_axis[0])
    gaussian_width = (vel_chan**2 - channel_in**2) ** 0.5 / channel_in / fwhm_factor

    smcube = cube_inter.spectral_smooth(Gaussian1DKernel(gaussian_width.value))
    new_cube = smcube.spectral_interpolate(new_axis, suppress_smooth_warning=True)
    return new_cube
