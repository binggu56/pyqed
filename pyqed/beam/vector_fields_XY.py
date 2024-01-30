# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module generates Vector_field_XY class.

It is required also for generating masks and fields.
The main atributes are:
    * self.Ex - x component of electric field
    * self.Ey - y component of electric field
    * self.x - x positions of the field
    * self.wavelength - wavelength of the incident field. The field is monocromatic
    * self.x - x positions of the field
    * self.y - y positions of the field
    * self.wavelength - wavdelength of the incident field. The field is monochromatic
    * self.Ex - x component of electric field. Equal size to x * y. complex field
    * self.Ey - y component of electric field. Equal size to x * y. complex field
    * self.X (numpy.array): equal size to x * y. complex field
    * self.Y (numpy.array): equal size to x * y. complex field
    * self.quality (float): quality of RS algorithm
    * self.info (str): description of data
    * self.type (str): Class of the field
    * self.date (str): date when performed


The magnitude is related to microns: `micron = 1.`

*Class for XY vector fields*

*Definition of a scalar field*
    * add, substract fields
    * save, load data, clean, get, normalize
    * cut_resample
    * appy_mask

*Vector parameters*
    * polarization_states
    * polarization_ellipse

*Propagation*
    * RS - Rayleigh Sommerfeld

*Drawing functions*
    * draw: intensity, intensities, phases, fields, stokes, param_ellipse, ellipses

"""

from matplotlib import rcParams
from numpy import cos, linspace, shape, sin, sqrt, zeros
from scipy.fftpack import fft2, fftshift, ifft2
from scipy.interpolate import RectBivariateSpline

from . import degrees, eps, mm, np, plt
from .config import CONF_DRAWING
from .scalar_fields_X import Scalar_field_X
from .scalar_fields_XY import Scalar_field_XY
from .scalar_masks_XY import Scalar_mask_XY
from .utils_common import load_data_common, save_data_common
from .utils_drawing import normalize_draw, reduce_matrix_size
from .utils_math import get_edges, get_k, ndgrid, nearest, rotate_image, Bluestein_dft_xy

percentage_intensity = CONF_DRAWING['percentage_intensity']


class Vector_field_XY(object):
    """Class for vectorial fields.

    Parameters:
        x (numpy.array): linear array with equidistant positions. The number of data is preferibly 2**n.
        y (numpy.array): linear array with equidistant positions. The number of data is preferibly 2**n.
        wavelength (float): wavelength of the incident field
        info (str): String with info about the simulation

    Attributes:
        self.x (numpy.array): linear array with equidistant positions. The number of data is preferibly 2**n.
        self.y (numpy.array): linear array with equidistant positions. The number of data is preferibly 2**n.
        self.wavelength (float): wavelength of the incident field.
        self.Ex (numpy.array): Electric_x field
        self.Ey (numpy.array): Electric_y field
    """

    def __init__(self, x, y, wavelength, info=''):
        self.x = x
        self.y = y
        self.wavelength = wavelength  # la longitud de onda
        self.X, self.Y = np.meshgrid(x, y)

        self.Ex = np.zeros_like(self.X, dtype=complex)
        self.Ey = np.zeros_like(self.X, dtype=complex)
        self.Ez = np.zeros_like(self.X, dtype=complex)

        self.reduce_matrix = 'standard'  # 'None, 'standard', (5,5)
        self._type = 'Vector_field_XY'

    def __str__(self):
        """Represents data from class."""

        print('length x: ', self.x.shape, '- xmin: ', self.x[0], '- xmax: ',
              self.x[-1])
        print('length y: ', self.y.shape, '- ymin: ', self.y[0], '- ymax: ',
              self.y[-1])
        print('size Ex: ', self.Ex.shape)

    def __add__(self, other, kind='standard'):
        """adds two Vector_field_XY. For example two light sources or two masks

        Parameters:
            other (Vector_field_XY): 2nd field to add
            kind (str): instruction how to add the fields:

        Returns:
            Vector_field_XY: `E3 = E1 + E2`
        """

        EM = Vector_field_XY(self.x, self.y, self.wavelength)

        if kind == 'standard':
            EM.Ex = self.Ex + other.Ex
            EM.Ey = self.Ey + other.Ey

        return EM

    def __rotate__(self, angle, position=None):
        """Rotation of X,Y with respect to position

        Parameters:
            angle (float): angle to rotate, in radians
            position (float, float): position of center of rotation
        """

        if position is None:
            x0 = (self.x[-1] + self.x[0]) / 2
            y0 = (self.y[-1] + self.y[0]) / 2
        else:
            x0, y0 = position

        Xrot = (self.X - x0) * cos(angle) + (self.Y - y0) * sin(angle)
        Yrot = -(self.X - x0) * sin(angle) + (self.Y - y0) * cos(angle)
        return Xrot, Yrot

    def save_data(self, filename, add_name='', description='', verbose=False):
        """Common save data function to be used in all the modules.
        The methods included are: npz, matlab


        Parameters:
            filename (str): filename
            add_name= (str): sufix to the name, if 'date' includes a date
            description (str): text to be stored in the dictionary to save.
            verbose (bool): If verbose prints filename.

        Returns:
            (str): filename. If False, file could not be saved.
        """
        try:
            final_filename = save_data_common(self, filename, add_name,
                                              description, verbose)
            return final_filename
        except:
            return False

    def load_data(self, filename, verbose=False):
        """Load data from a file to a Vector_field_XY.
            The methods included are: npz, matlab

        Parameters:
            filename (str): filename
            verbose (bool): shows data process by screen
        """
        dict0 = load_data_common(self, filename)

        if dict0 is not None:
            if isinstance(dict0, dict):
                self.__dict__ = dict0
            else:
                raise Exception('no dictionary in load_data')

        if verbose is True:
            print(dict0.keys())

    def clear(self):
        """simple - removes the field: self.E=0 """

        self.Ex = np.zeros_like(self.Ex, dtype=complex)
        self.Ey = np.zeros_like(self.Ex, dtype=complex)

    def duplicate(self, clear=False):
        """Duplicates the instance"""
        new_field = copy.deepcopy(self)
        if clear is True:
            new_field.clear_field()
        return new_field

    def get(self, kind='fields', is_matrix=True):
        """Takes the vector field and divide in Scalar_field_XY

        Parameters:
            kind (str): 'fields', 'intensity', 'intensities', 'phases', 'stokes', 'params_ellipse'

        Returns:
            Scalar_field_XY: (Ex, Ey),
        """

        Ex_r = self.Ex
        Ey_r = self.Ey
        Ez_r = self.Ez

        if kind == 'fields':
            if is_matrix:
                return self.Ex, self.Ey, self.Ez

            else:
                Ex = Scalar_field_XY(x=self.x,
                                     y=self.y,
                                     wavelength=self.wavelength)
                Ex.u = Ex_r
                Ey = Scalar_field_XY(x=self.x,
                                     y=self.y,
                                     wavelength=self.wavelength)
                Ey.u = Ey_r
                Ez = Scalar_field_XY(x=self.x,
                                     y=self.y,
                                     wavelength=self.wavelength)
                Ez.u = Ez_r
                return Ex, Ey, Ez

        elif kind == 'intensity':
            intensity = np.abs(Ex_r)**2 + np.abs(Ey_r)**2 + np.abs(Ez_r)**2

            if is_matrix:
                return intensity

            else:
                Intensity = Scalar_field_XY(x=self.x,
                                            y=self.y,
                                            wavelength=self.wavelength)
                Intensity.u = np.sqrt(intensity)

                return Intensity

        elif kind == 'intensities':
            intensity_x = np.abs(Ex_r)**2
            intensity_y = np.abs(Ey_r)**2
            intensity_z = np.abs(Ez_r)**2
            return intensity_x, intensity_y, intensity_z

        elif kind == 'phases':
            phase_x = np.angle(Ex_r)
            phase_y = np.angle(Ey_r)
            phase_z = np.angle(Ez_r)

            if is_matrix:
                return phase_x, phase_y, phase_z
            else:
                Ex = Scalar_field_XY(x=self.x,
                                     y=self.y,
                                     wavelength=self.wavelength)
                Ex.u = np.exp(1j * phase_x)
                Ey = Scalar_field_XY(x=self.x,
                                     y=self.y,
                                     wavelength=self.wavelength)
                Ey.u = np.exp(1j * phase_y)
                Ez = Scalar_field_XY(x=self.x,
                                     y=self.y,
                                     wavelength=self.wavelength)
                Ez.u = np.exp(1j * phase_z)
                return Ex, Ey, Ez

        elif kind == 'stokes':
            # S0, S1, S2, S3
            return self.polarization_states(matrix=True)

        elif kind == 'params_ellipse':
            # A, B, theta, h
            return self.polarization_ellipse(pol_state=None, matrix=True)

        else:
            print("The parameter '{}'' in .get(kind='') is wrong".format(kind))

    def pupil(self, r0=None, radius=None, angle=0 * degrees):
        """place a pupil in the field. If r0 or radius are None, they are computed using the x,y parameters.

        Parameters:
            r0 (float, float): center of circle/ellipse
            radius (float, float) or (float): radius of circle/ellipse
            angle (float): angle of rotation in radians

        Example:

            pupil(r0=(0 * um, 0 * um), radius=(250 * \
                   um, 125 * um), angle=0 * degrees)
        """

        if r0 is None:
            x0 = (self.x[-1] + self.x[0]) / 2
            y0 = (self.y[-1] + self.y[0]) / 2
            r0 = (x0, y0)

        if radius is None:
            radiusx = (self.x[-1] - self.x[0]) / 2
            radiusy = (self.y[-1] - self.y[0]) / 2
            radius = (radiusx, radiusy)

        x0, y0 = r0

        if isinstance(radius, (float, int, complex)):
            radiusx, radiusy = (radius, radius)
        else:
            radiusx, radiusy = radius

        # Rotacion del circula/elipse
        Xrot, Yrot = self.__rotate__(angle, (x0, y0))

        # Definicion de la transmitancia
        pupil0 = zeros(shape(self.X))
        ipasa = (Xrot)**2 / (radiusx + 1e-15)**2 + (Yrot)**2 / (radiusy**2 + 1e-15) < 1
        pupil0[ipasa] = 1
        self.Ex = self.Ex * pupil0
        self.Ey = self.Ey * pupil0
        self.Ez = self.Ez * pupil0

    def apply_mask(self, u):
        """Multiply field by binary scalar mask: self.Ex = self.Ex * u.u

        Parameters:
           u (Scalar_mask_XY): mask
         """
        self.Ex = self.Ex * u.u
        self.Ey = self.Ey * u.u
        self.Ez = self.Ez * u.u

    def intensity(self):
        """"Returns intensity.
        """
        intensity = np.abs(self.Ex)**2 + np.abs(self.Ey)**2 + np.abs(self.Ez)**2

        return intensity

    def RS(self,
           z=10 * mm,
           n=1,
           new_field=True,
           amplification=(1, 1),
           verbose=False):
        """Fast-Fourier-Transform  method for numerical integration of diffraction Rayleigh-Sommerfeld formula. `Thin Element Approximation` is considered for determining the field just after the mask: :math:`\mathbf{E}_{0}(\zeta,\eta)=t(\zeta,\eta)\mathbf{E}_{inc}(\zeta,\eta)` Is we have a field of size N*M, the result of propagation is also a field N*M. Nevertheless, there is a parameter `amplification` which allows us to determine the field in greater observation planes (jN)x(jM).


        Parameters:
            z (float): distance to observation plane.
                if z<0 inverse propagation is executed
            n (float): refraction index
            new_field (bool): if False the computation goes to self.u
                              if True a new instance is produced
            amplification (int, int): number of frames in x and y direction

            verbose (bool): if True it writes to shell. Not implemented yet

        Returns:
            if New_field is True: Scalar_field_X
            else None


        Note:
            One adventage of this approach is that it returns a quality parameter: if self.quality>1, propagation is right.

        References:
            From Applied Optics vol 45 num 6 pp. 1102-1110 (2006)
        """

        e0x, e0y, _ = self.get()

        # estas son las components justo en la posicion pedida
        Ex = e0x.RS(z=z,
                    n=n,
                    new_field=True,
                    kind='z',
                    amplification=amplification)
        Ey = e0y.RS(z=z,
                    n=n,
                    new_field=True,
                    kind='z',
                    amplification=amplification,
                    verbose=verbose)

        if new_field is True:
            EM = Vector_field_XY(Ex.x, Ex.y, self.wavelength)
            EM.Ex = Ex.u
            EM.Ey = Ey.u
            EM.Ez = np.zeros_like(EM.X)
            EM.x = Ex.x
            EM.y = Ex.y
            return EM

        else:
            self.Ex = Ex.u
            self.Ey = Ey.u
            self.Ez = np.zeros_like(EM.X)
            self.x = Ex.x
            self.y = Ex.y



    def VFFT(self,
             radius,
             focal,
             n=1,
             new_field=False,
             shift=True,
             remove0=True,
             matrix=False,
             has_draw=False):
        """Vector Fast Fourier Transform (FFT) of the field.

        The focusing system, shown schematically in Fig. 1 is modelled by a high NA, aberration-free, aplanatic lens obeying the sine condition,
        having a focal length fand collecting light under a convergence angle theta_max.
        Denoting the refractive index of the medium in the focal region with n, the NA of the lens can be written as NA= n sin theta_max.
        The polarization changes on the lens surfaces described by the Fresnel formulae have been neglected.

        Ei = (Eix, Eiy, Eiz) is the local electric field vector.

        Parameters:
            radius (float): radius of lens
            focal (float): focal
            n (float): refraction index
            matrix (bool):  if True only matrix is returned. if False, returns Scalar_field_X.
            new_field (bool): if True returns Vector_field_XY, else it puts in self.
            shift (bool):  if True, fftshift is performed.
            remove0 (bool): if True, central point is removed.
            has_draw (bool): if True draw the field.

        Returns:
            (np.array or vector_fields_XY or None): FFT of the input field.

        Reference:
            Jahn, Kornél, and Nándor Bokor. 2010. “Intensity Control of the Focal Spot by Vectorial Beam Shaping.” Optics Communications 283 (24): 4859–65. https://doi.org/10.1016/j.optcom.2010.07.030.


        TODO:
             some inconsistency in the radius of the circle lower than the size of the field.
        """
        from .vector_sources_XY import Vector_source_XY

        # dx = self.x[1] - self.x[0]
        # dy = self.y[1] - self.y[0]
        num_x, num_y = self.X.shape

        # numerical aperture
        sin_theta_max = radius / np.sqrt(radius**2 + focal**2)
        # NA = n * sin_theta_max

        r = np.sqrt(self.X**2 + self.Y**2)
        phi = np.arctan2(self.Y, self.X)
        theta = r / focal

        u = self.X / radius
        v = self.Y / radius

        # X_obs = sin_theta_max * self.X / self.wavelength
        # Y_obs = sin_theta_max * self.Y / self.wavelength

        circle_mask = Scalar_mask_XY(self.x, self.y, self.wavelength)
        circle_mask.circle(r0=(0, 0), radius=radius)

        self.mask_circle(r0=(0., 0.), radius=radius)

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        apodization_factor = np.sqrt(np.abs(np.cos(theta)))

        G = 1 / sqrt(np.abs(1 - sin_theta_max**2 * (u**2 + v**2)))
        G = G * circle_mask.u
        G = np.real(G)

        M00 = cos_phi**2 * cos_theta + sin_phi**2
        M01 = sin_phi * cos_phi * cos_theta - sin_phi * cos_phi
        M02 = -sin_theta * cos_phi

        M10 = sin_phi * cos_phi * cos_theta - sin_phi * cos_phi
        M11 = sin_phi**2 * cos_theta + cos_phi**2
        M12 = -sin_theta * sin_phi

        M20 = sin_theta * cos_phi
        M21 = sin_theta * sin_phi
        M22 = cos_theta

        Eix = self.Ex
        Eiy = self.Ey
        Eiz = self.Ez

        E0x = M00 * Eix + M01 * Eiy + M02 * Eiz
        E0y = M10 * Eix + M11 * Eiy + M12 * Eiz
        E0z = M20 * Eix + M21 * Eiy + M22 * Eiz

        factor = -(1j * sin_theta_max**2 / (focal * self.wavelength))
        Ep_x = fft2(apodization_factor * G * E0x)
        Ep_y = fft2(apodization_factor * G * E0y)
        Ep_z = fft2(apodization_factor * G * E0z)

        if remove0:
            Ep_x[0, 0] = 0
            Ep_y[0, 0] = 0
            # Ep_z[0,0] = 0

        if shift:
            Ep_x = fftshift(Ep_x)
            Ep_y = fftshift(Ep_y)
            Ep_z = fftshift(Ep_z)

        Esx = factor * Ep_x
        Esy = factor * Ep_y
        Esz = factor * Ep_z

        if matrix is True:
            return np.stack((Esx, Esy, Esz), axis=2)

        num_x = self.x.size
        delta_x = self.x[1] - self.x[0]
        freq_nyquist_x = 1 / (2 * delta_x)

        kx = linspace(-freq_nyquist_x, freq_nyquist_x, num_x) * focal
        num_y = self.y.size
        delta_y = self.y[1] - self.y[0]
        freq_nyquist_y = 1 / (2 * delta_y)
        ky = linspace(-freq_nyquist_y, freq_nyquist_y, num_y) * focal

        if new_field is True:
            field_output = Vector_source_XY(self.x, self.y, self.wavelength)
            field_output.x = kx
            field_output.y = ky

            field_output.X, field_output.Y = ndgrid(field_output.x,
                                                    field_output.y)
            field_output.Ex = Esx
            field_output.Ey = Esy
            field_output.Ez = Esz

            if has_draw:
                field_output.draw(kind='intensities')

            return field_output

        else:
            self.Ex = Esx
            self.Ey = Esy
            self.Ez = Esz
            self.x = kx
            self.y = ky
            self.X, self.Y = ndgrid(self.x, self.y)

            if has_draw:
                self.draw(kind='intensities')

    def IVFFT(self,
              radius,
              focal,
              n=1,
              new_field=False,
              matrix=False,
              has_draw=False):
        """Inverse Vector Fast Fourier Transform (FFT) of the field.

        The focusing system, shown schematically in Fig. 1 is modelled by a high NA, aberration-free, aplanatic lens obeying the sine condition,
        having a focal length fand collecting light under a convergence angle theta_max.
        Denoting the refractive index of the medium in the focal region with n, the NA of the lens can be written as NA= n sin theta_max.
        The polarization changes on the lens surfaces described by the Fresnel formulae have been neglected.

        Ei = (Eix, Eiy, Eiz) is the local electric field vector.

        Parameters:
            radius (float): radius of lens
            focal (float): focal
            n (float): refraction index
            matrix (bool):  if True only matrix is returned. if False, returns Scalar_field_X.
            new_field (bool): if True returns Vector_field_XY, else it puts in self.
            has_draw (bool): if True draw the field.

        Returns:
            (np.array or vector_fields_XY or None): FFT of the input field.

        Reference:
            Jahn, Kornél, and Nándor Bokor. 2010. “Intensity Control of the Focal Spot by Vectorial Beam Shaping.” Optics Communications 283 (24): 4859–65. https://doi.org/10.1016/j.optcom.2010.07.030.


        TODO:
            radius of the circle lower than the size of the field.
        """

        from diffractio.vector_sources_XY import \
            Vector_source_XY

        # numerical aperture
        sin_theta_max = radius / np.sqrt(radius**2 + focal**2)
        # NA = n * sin_theta_max

        # dx = self.x[1] - self.x[0]
        # dy = self.y[1] - self.y[0]
        num_x, num_y = self.X.shape

        r = np.sqrt(self.X**2 + self.Y**2)
        phi = np.arctan2(self.Y, self.X)
        theta = r / focal

        u = self.X / radius
        v = self.Y / radius

        # X_obs = sin_theta_max * self.X / self.wavelength
        # Y_obs = sin_theta_max * self.Y / self.wavelength

        circle_mask = Scalar_mask_XY(self.x, self.y, self.wavelength)
        circle_mask.circle(r0=(0, 0), radius=radius)

        self.mask_circle(r0=(0., 0.), radius=radius)

        cos_theta = np.cos(theta)
        sin_theta = -np.sin(theta)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        #  apodization_factor = np.sqrt(np.abs(np.cos(theta)))

        G = sqrt(np.abs(1 - sin_theta_max**2 * (u**2 + v**2)))
        G = G * circle_mask.u
        G = np.real(G)

        M00 = cos_phi**2 * cos_theta + sin_phi**2
        M01 = sin_phi * cos_phi * cos_theta - sin_phi * cos_phi
        M02 = -sin_theta * cos_phi

        M10 = sin_phi * cos_phi * cos_theta - sin_phi * cos_phi
        M11 = sin_phi**2 * cos_theta + cos_phi**2
        M12 = -sin_theta * sin_phi

        M20 = sin_theta * cos_phi
        M21 = sin_theta * sin_phi
        M22 = cos_theta

        Eix = self.Ex
        Eiy = self.Ey
        Eiz = self.Ez

        factor = -(1j * sin_theta_max**2 / (focal * self.wavelength))**(-1)

        Esx = factor * ifft2(Eix * G)
        Esy = factor * ifft2(Eiy * G)
        Esz = factor * ifft2(Eiz * G)

        Esx = factor * ifft2(Eix * G)
        Esy = factor * ifft2(Eiy * G)
        Esz = factor * ifft2(Eiz * G)

        E0x = M00 * Esx + M01 * Esy + M02 * Esz
        E0y = M10 * Esx + M11 * Esy + M12 * Esz
        E0z = M20 * Esx + M21 * Esy + M22 * Esz

        Esx = E0x
        Esy = E0y
        Esz = E0z

        if matrix is True:
            return np.stack((Esx, Esy, Esz), axis=2)

        num_x = self.x.size
        delta_x = self.x[1] - self.x[0]
        freq_nyquist_x = 1 / (2 * delta_x)

        kx = linspace(-freq_nyquist_x, freq_nyquist_x, num_x) * focal
        num_y = self.y.size
        delta_y = self.y[1] - self.y[0]
        freq_nyquist_y = 1 / (2 * delta_y)
        ky = linspace(-freq_nyquist_y, freq_nyquist_y, num_y) * focal

        if new_field is True:
            field_output = Vector_source_XY(self.x, self.y, self.wavelength)
            field_output.x = kx
            field_output.y = ky

            field_output.X, field_output.Y = ndgrid(field_output.x,
                                                    field_output.y)
            field_output.Ex = Esx
            field_output.Ey = Esy
            field_output.Ez = Esz

            if has_draw:
                field_output.draw(kind='intensities')

            return field_output

        else:
            self.Ex = Esx
            self.Ey = Esy
            self.Ez = Esz
            self.x = kx
            self.y = ky
            self.X, self.Y = ndgrid(self.x, self.y)

            if has_draw:
                self.draw(kind='intensities')

    def VRS(self, z, n=1, new_field=True, verbose=False, amplification=(1, 1)):
        """Fast-Fourier-Transform  method for numerical integration of diffraction Vector Rayleigh-Sommerfeld formula.


        Parameters:
            z (float): distance to observation plane.
                if z<0 inverse propagation is executed
            n (float): refraction index
            new_field (bool): if False the computation goes to self.u
                              if True a new instance is produced


            verbose (bool): if True it writes to shell. Not implemented yet

        Returns:
            if New_field is True: Scalar_field_X
            else None

        References:

            H. Ye, C.-W. Qiu, K. Huang, J. Teng, B. Luk’yanchuk, y S. P. Yeo, «Creation of a longitudinally polarized subwavelength hotspot with an ultra-thin planar lens: vectorial Rayleigh–Sommerfeld method», Laser Phys. Lett., vol. 10, n.º 6, p. 065004, jun. 2013.

            DOI: 10.1088/1612-2011/10/6/065004

            http://stacks.iop.org/1612-202X/10/i=6/a=065004?key=crossref.890761f053b56d7a9eeb8fc6da4d9b4e
        """

        e0x, e0y, _ = self.get(is_matrix=False)

        e0z = Scalar_field_XY(x=self.x, y=self.y, wavelength=self.wavelength)
        e0z.u = 0

        # estas son las components justo en la posicion pedida
        Ex = e0x.RS(z=z,
                    n=n,
                    new_field=True,
                    kind='z',
                    verbose=verbose,
                    amplification=amplification)
        Ey = e0y.RS(z=z,
                    n=n,
                    new_field=True,
                    kind='z',
                    verbose=verbose,
                    amplification=amplification)

        r = np.sqrt(self.X**2 + self.Y**2 + z**2)
        e0z.u = e0x.u * self.X / r + e0y.u * self.Y / r
        Ez = e0z.RS(z=z,
                    n=n,
                    new_field=True,
                    kind='0',
                    verbose=verbose,
                    amplification=amplification)

        if new_field is True:
            EM = Vector_field_XY(self.x, self.y, self.wavelength)
            EM.Ex = Ex.u
            EM.Ey = Ey.u
            EM.Ez = Ez.u
            return EM

        else:
            self.Ex = Ex.u
            self.Ey = Ey.u
            self.Ez = Ez.u


    def CZT(self, z, xout, yout):
        """_summary_ from XY to XY

        previous: Scalar_Bluestein_XY

        Parameters:
            z (float): diffraction distance
            xout (np.array): x array with positions of the output plane
            yout (np.array): y array with positions of the output plane


        Returns:
            gout: Complex amplitude of the outgoing light beam
            delta_out: pixel size of the outgoing light beam
        """

        e0x, e0y, _ = self.get(is_matrix=True)


        k = 2 * np.pi / self.wavelength

        if isinstance(xout, (float, int)):
            numx_out = 2
            xout = np.array((xout, xout + 0.1))
            remove_x = True
        else:
            numx_out = len(xout)
            remove_x = False

        if isinstance(yout, (float, int)):
            numy_out = 2
            yout = np.array((yout, yout + 0.1))
            remove_y = True

        else:
            numy_out = len(yout)
            remove_y = False

        xstart = xout[0]
        xend = xout[-1]
        ystart = yout[0]
        yend = yout[-1]

        delta_x_in = self.x[1] - self.x[0]

        delta_out = np.zeros(2)
        if numx_out > 1:
            delta_out[0] = (xend - xstart) / (numx_out - 1)
        if numy_out > 1:
            delta_out[1] = (yend - ystart) / (numy_out - 1)

        Xout, Yout = np.meshgrid(xout, yout)

        # Fresnel
        # F0 = exp(1j * k * z) / (1j * self.wavelength * z) * exp(
        #     1j * k / 2 / z * (Xout**2 + Yout**2))
        # F = exp(1j * k / 2 / z * (self.X**2 + self.Y**2))


        R = sqrt(Xout**2 + Yout**2 + z**2)
        F0 = 1 / (2 * np.pi) * np.exp(1.j * k * R) * z / R**2 * (1 / R - 1.j * k)

        R = sqrt(self.X**2 + self.Y**2 + z**2)
        F = 1 / (2 * np.pi) * np.exp(1.j * k * R) * z / R**2 * (1 / R - 1.j * k)

        r = np.sqrt(self.X**2 + self.Y**2 + z**2)
        e0z = e0x * self.X / r + e0y * self.Y / r
        e0z = e0z * z/r

        Ex0 = e0x * F
        Ey0 = e0y * F
        Ez0 = e0z * F

        # using Bluestein method to calculate the complex amplitude of the outgoing light beam

        # one-dimensional FFT in one direction
        fs = self.wavelength * z / delta_x_in  # dimension of the imaging plane

        fy1 = ystart + fs / 2
        fy2 = yend + fs / 2
        Ex0 = Bluestein_dft_xy(Ex0, fy1, fy2, fs, numy_out)
        Ey0 = Bluestein_dft_xy(Ey0, fy1, fy2, fs, numy_out)
        Ez0 = Bluestein_dft_xy(Ez0, fy1, fy2, fs, numy_out)

        # one-dimensional FFT in the other direction
        fx1 = xstart + fs / 2
        fx2 = xend + fs / 2
        Ex0 = Bluestein_dft_xy(Ex0, fx1, fx2, fs, numx_out)
        Ey0 = Bluestein_dft_xy(Ey0, fx1, fx2, fs, numx_out)
        Ez0 = Bluestein_dft_xy(Ez0, fx1, fx2, fs, numx_out)

        # obtain the complex amplitude of the outgoing light beam
        Ex0 = F0 * Ex0
        Ey0 = F0 * Ey0
        Ez0 = F0 * Ez0

        if remove_x is True:
            # Ex0 = Ex0[:, 0]
            # Ey0 = Ey0[:, 0]
            # Ez0 = Ez0[:, 0]
            xout = np.array((xout[0], ))
            ex_out = Scalar_field_X(yout, self.wavelength)
            ey_out = Scalar_field_X(yout, self.wavelength)
            ez_out = Scalar_field_X(yout, self.wavelength)
            ex_out.u = Ex0[:,0]
            ey_out.u = Ey0[:,0]
            ez_out.u = Ez0[:,0]
            return ex_out, ey_out, ez_out

        if remove_y is True:
            # Ex0 = Ex0[:, 0]
            # Ey0 = Ey0[:, 0]
            # Ez0 = Ey0[:, 0]
            yout = np.array((yout[0], ))
            ex_out = Scalar_field_X(xout, self.wavelength)
            ey_out = Scalar_field_X(xout, self.wavelength)
            ez_out = Scalar_field_X(xout, self.wavelength)
            ex_out.u = Ex0[0,:]
            ey_out.u = Ey0[0,:]
            ez_out.u = Ez0[0,:]
            return ex_out, ey_out, ez_out

        E_out = Vector_field_XY(xout, yout, self.wavelength)
        E_out.Ex = Ex0
        E_out.Ey = Ey0
        E_out.Ez = Ez0
        return E_out

    def polarization_states(self, matrix=False):
        """returns the Stokes parameters

        Parameters:
            Matrix (bool): if True returns Matrix, else Scalar_field_XY

        Returns:
            S0,S1,S2,S3 images for Matrix=True
            S0,S1,S2,S3  for Matrix=False
        """

        I = np.abs(self.Ex)**2 + np.abs(self.Ey)**2
        Q = np.abs(self.Ex)**2 - np.abs(self.Ey)**2
        U = 2 * np.real(self.Ex * np.conjugate(self.Ey))
        V = 2 * np.imag(self.Ex * np.conjugate(self.Ey))

        if matrix is True:
            return I, Q, U, V
        else:
            CI = Scalar_field_XY(x=self.x,
                                 y=self.y,
                                 wavelength=self.wavelength)
            CQ = Scalar_field_XY(x=self.x,
                                 y=self.y,
                                 wavelength=self.wavelength)
            CU = Scalar_field_XY(x=self.x,
                                 y=self.y,
                                 wavelength=self.wavelength)
            CV = Scalar_field_XY(x=self.x,
                                 y=self.y,
                                 wavelength=self.wavelength)

            CI.u = I
            CQ.u = Q
            CU.u = U
            CV.u = V

            return CI, CQ, CU, CV

    def polarization_ellipse(self, pol_state=None, matrix=False):
        """returns A, B, theta, h polarization parameter of elipses

        Parameters:
            pol_state (None or (I, Q, U, V) ): Polarization state previously computed
            Matrix (bool): if True returns Matrix, else Scalar_field_XY

        Returns:
            A, B, theta, h for Matrix=True
            CA, CB, Ctheta, Ch for Matrix=False
        """
        if pol_state is None:
            I, Q, U, V = self.polarization_states(matrix=True)
        else:
            I, Q, U, V = pol_state
            I = I.u
            Q = Q.u
            U = U.u
            V = V.u

        Ip = np.sqrt(Q**2 + U**2 + V**2)
        L = Q + 1.j * U + eps

        A = np.real(np.sqrt(0.5 * (Ip + np.abs(L) + eps)))
        B = np.real(np.sqrt(0.5 * (Ip - np.abs(L) + eps)))
        theta = 0.5 * np.angle(L)
        h = np.sign(V + eps)

        if matrix is True:
            return A, B, theta, h
        else:
            CA = Scalar_field_XY(x=self.x,
                                 y=self.y,
                                 wavelength=self.wavelength)
            CB = Scalar_field_XY(x=self.x,
                                 y=self.y,
                                 wavelength=self.wavelength)
            Ctheta = Scalar_field_XY(x=self.x,
                                     y=self.y,
                                     wavelength=self.wavelength)
            Ch = Scalar_field_XY(x=self.x,
                                 y=self.y,
                                 wavelength=self.wavelength)

            CA.u = A
            CB.u = B
            Ctheta.u = theta
            Ch.u = h
            return (CA, CB, Ctheta, Ch)

    def normalize(self):
        """Normalizes the field"""
        max_amplitude = np.sqrt(
            np.abs(self.Ex)**2 + np.abs(self.Ey)**2 +
            np.abs(self.Ez)**2).max()

        self.Ex = self.Ex / max_amplitude
        self.Ey = self.Ey / max_amplitude
        self.Ez = self.Ez / max_amplitude

    def cut_resample(self,
                     x_limits='',
                     y_limits='',
                     num_points=[],
                     new_field=False,
                     interp_kind=(3, 1)):
        """Cuts the field to the range (x0,x1). (y0,y1). If one of this x0,x1 positions is out of the self.x range it do nothing. It is also valid for resampling the field, just write x0,x1 as the limits of self.x

        Parameters:
            x_limits (float,float): (x0,x1) starting and final points to cut. if '' - takes the current limit x[0] and x[-1]
            y_limits (float,float): (y0,y1) - starting and final points to cut. if '' - takes the current limit y[0] and y[-1]
            num_points (int): it resamples x, y and u. [],'',0,None -> it leave the points as it is
            new_field (bool): it returns a new Scalar_field_XY
            interp_kind (int): numbers between 1 and 5
        """
        if x_limits == '':
            # used only for resampling
            x0 = self.x[0]
            x1 = self.x[-1]
        else:
            x0, x1 = x_limits

        if y_limits == '':
            # used only for resampling
            y0 = self.y[0]
            y1 = self.y[-1]
        else:
            y0, y1 = y_limits

        if x0 < self.x[0]:
            x0 = self.x[0]
        if x1 > self.x[-1]:
            x1 = self.x[-1]

        if y0 < self.y[0]:
            y0 = self.y[0]
        if y1 > self.y[-1]:
            y1 = self.y[-1]

        i_x0, _, _ = nearest(self.x, x0)
        i_x1, _, _ = nearest(self.x, x1)
        # new_num_points = i_x1 - i_x0
        i_y0, _, _ = nearest(self.y, y0)
        i_y1, _, _ = nearest(self.y, y1)

        kxu, kxn = interp_kind

        if num_points not in ([], '', 0, None):
            num_points_x, num_points_y = num_points
            x_new = np.linspace(x0, x1, num_points_x)
            y_new = np.linspace(y0, y1, num_points_y)
            X_new, Y_new = np.meshgrid(x_new, y_new)

            f_interp_abs_x = RectBivariateSpline(self.x,
                                                 self.y,
                                                 np.abs(self.Ex),
                                                 kx=kxu,
                                                 ky=kxu,
                                                 s=0)
            f_interp_phase_x = RectBivariateSpline(self.x,
                                                   self.y,
                                                   np.angle(self.Ex),
                                                   kx=kxu,
                                                   ky=kxu,
                                                   s=0)

            f_interp_abs_y = RectBivariateSpline(self.x,
                                                 self.y,
                                                 np.abs(self.Ey),
                                                 kx=kxu,
                                                 ky=kxu,
                                                 s=0)
            f_interp_phase_y = RectBivariateSpline(self.x,
                                                   self.y,
                                                   np.angle(self.Ey),
                                                   kx=kxu,
                                                   ky=kxu,
                                                   s=0)

            f_interp_abs_z = RectBivariateSpline(self.x,
                                                 self.y,
                                                 np.abs(self.Ez),
                                                 kx=kxu,
                                                 ky=kxu,
                                                 s=0)
            f_interp_phase_z = RectBivariateSpline(self.x,
                                                   self.y,
                                                   np.angle(self.Ez),
                                                   kx=kxu,
                                                   ky=kxu,
                                                   s=0)

            Ex_new_abs = f_interp_abs_x(x_new, y_new)
            Ex_new_phase = f_interp_phase_x(x_new, y_new)
            Ex_new = Ex_new_abs * np.exp(1j * Ex_new_phase)

            Ey_new_abs = f_interp_abs_y(x_new, y_new)
            Ey_new_phase = f_interp_phase_y(x_new, y_new)
            Ey_new = Ey_new_abs * np.exp(1j * Ey_new_phase)

            Ez_new_abs = f_interp_abs_z(x_new, y_new)
            Ez_new_phase = f_interp_phase_z(x_new, y_new)
            Ez_new = Ez_new_abs * np.exp(1j * Ez_new_phase)

        else:
            i_s = slice(i_x0, i_x1)
            j_s = slice(i_y0, i_y1)
            x_new = self.x[i_s]
            y_new = self.y[j_s]
            X_new, Y_new = np.meshgrid(x_new, y_new)
            Ex_new = self.Ex[i_s, j_s]
            Ey_new = self.Ey[i_s, j_s]
            Ez_new = self.Ez[i_s, j_s]

        if new_field is False:
            self.x = x_new
            self.y = y_new
            self.Ex = Ex_new
            self.Ey = Ey_new
            self.Ez = Ez_new
            self.X = X_new
            self.Y = Y_new

        elif new_field is True:
            field = Vector_field_XY(x=x_new,
                                    y=y_new,
                                    wavelength=self.wavelength)
            field.Ex = Ex_new
            field.Ey = Ey_new
            field.Ez = Ez_new
            return field

    def draw(self,
             kind='intensity',
             logarithm=0,
             normalize=False,
             cut_value=None,
             num_ellipses=(11, 11),
             amplification=0.5,
             filename='',
             draw=True,
             only_image=False,
             **kwargs):
        """Draws electromagnetic field

        Parameters:
            kind (str):  'intensity', 'intensities', intensities_rz, 'phases', fields', 'stokes', 'param_ellipse', 'ellipses'
            logarithm (float): If >0, intensity is scaled in logarithm
            normalize (bool): If True, max(intensity)=1
            cut_value (float): If not None, cuts the maximum intensity to this value
            num_ellipses (int): number of ellipses for parameters_ellipse
            amplification (float): amplification of ellipses
            filename (str): if not '' stores drawing in file,

        """

        if draw is True:
            if kind == 'intensity':
                id_fig = self.__draw_intensity__(logarithm, normalize,
                                                 cut_value, only_image,
                                                 **kwargs)
            elif kind == 'intensities':
                id_fig = self.__draw_intensities__(logarithm, normalize,
                                                   cut_value, **kwargs)

            elif kind == 'intensities_rz':
                id_fig = self.__draw_intensities_rz__(logarithm, normalize,
                                                      cut_value, **kwargs)

            elif kind == 'phases':
                id_fig = self.__draw_phases__(**kwargs)

            elif kind == 'fields':
                id_fig = self.__draw_fields__(logarithm, normalize, cut_value,
                                              **kwargs)

            elif kind == 'stokes':
                id_fig = self.__draw_stokes__(logarithm, normalize, cut_value,
                                              **kwargs)

            elif kind == 'param_ellipse':
                id_fig = self.__draw_param_ellipse__(**kwargs)

            elif kind == 'ellipses':
                id_fig = self.__draw_ellipses__(logarithm, normalize,
                                                cut_value, num_ellipses,
                                                amplification, **kwargs)

            else:
                print("not good kind parameter in vector_fields_XY.draw()")
                id_fig = None

            if not filename == '':
                plt.savefig(filename,
                            dpi=300,
                            bbox_inches='tight',
                            pad_inches=0.1)

            return id_fig

    def __draw_intensity__(self,
                           logarithm,
                           normalize,
                           cut_value,
                           only_image=False,
                           color_intensity=CONF_DRAWING['color_intensity']):
        """Draws the intensity

        Parameters:
            logarithm (bool): If True, intensity is scaled in logarithm
            normalize (bool): If True, max(intensity)=1
            cut_value (float): If not None, cuts the maximum intensity to this value
        """

        intensity = self.get('intensity')

        intensity = reduce_matrix_size(self.reduce_matrix, self.x, self.y,
                                       intensity)

        intensity = normalize_draw(intensity, logarithm, normalize, cut_value)

        plt.figure()
        h1 = plt.subplot(1, 1, 1)
        __draw1__(self, intensity, color_intensity, "", only_image=only_image)
        plt.subplots_adjust(left=0,
                            bottom=0,
                            right=1,
                            top=1,
                            wspace=0.05,
                            hspace=0)
        plt.tight_layout()
        return h1

    def __draw_phases__(self, color_phase=CONF_DRAWING['color_phase']):
        """internal funcion: draws intensity X,Y.

        Parameters:
            logarithm (bool): If True, intensity is scaled in logarithm
            normalize (bool): If True, max(intensity)=1
            cut_value (float): If not None, cuts the maximum intensity to this value
        """

        Ex_r = reduce_matrix_size(self.reduce_matrix, self.x, self.y, self.Ex)
        Ey_r = reduce_matrix_size(self.reduce_matrix, self.x, self.y, self.Ey)
        Ez_r = reduce_matrix_size(self.reduce_matrix, self.x, self.y, self.Ez)
        tx, ty = rcParams['figure.figsize']

        # intensity1 = np.abs(Ex_r)**2
        # intensity2 = np.abs(Ey_r)**2
        intensity3 = np.abs(Ez_r)**2

        # intensity_max = np.max(
        #     (intensity1.max(), intensity2.max(), intensity3.max()))

        if intensity3.max() < 1e-15:

            plt.figure(figsize=(2 * tx, ty))

            h1 = plt.subplot(1, 2, 1)

            phase = np.angle(Ex_r)
            intensity = np.abs(Ex_r)**2

            phase[intensity < percentage_intensity * (intensity.max())] = 0

            __draw1__(self, phase / degrees, color_phase, "$\phi_x$")
            plt.clim(-180, 180)

            h2 = plt.subplot(1, 2, 2)

            phase = np.angle(Ey_r)
            intensity = np.abs(Ey_r)**2

            phase[intensity < percentage_intensity * (intensity.max())] = 0

            __draw1__(self, phase / degrees, color_phase, "$\phi_y$")
            plt.clim(-180, 180)

            plt.subplots_adjust(left=0,
                                bottom=0,
                                right=1,
                                top=1,
                                wspace=0.05,
                                hspace=0)
            plt.tight_layout()

            return h1, h2
        else:

            plt.figure(figsize=(3 * tx, ty))

            h1 = plt.subplot(1, 3, 1)

            phase = np.angle(Ex_r)
            intensity = np.abs(Ex_r)**2

            phase[intensity < percentage_intensity * (intensity.max())] = 0

            __draw1__(self, phase / degrees, color_phase, "$\phi_x$")
            plt.clim(-180, 180)

            h2 = plt.subplot(1, 3, 2)

            phase = np.angle(Ey_r)
            intensity = np.abs(Ey_r)**2

            phase[intensity < percentage_intensity * (intensity.max())] = 0

            __draw1__(self, phase / degrees, color_phase, "$\phi_y$")
            plt.clim(-180, 180)

            h3 = plt.subplot(1, 3, 3)

            phase = np.angle(Ez_r)
            intensity = np.abs(Ez_r)**2

            phase[intensity < percentage_intensity * (intensity.max())] = 0

            __draw1__(self, phase / degrees, color_phase, "$\phi_z$")
            plt.clim(-180, 180)

            plt.subplots_adjust(left=0,
                                bottom=0,
                                right=1,
                                top=1,
                                wspace=0.05,
                                hspace=0)
            plt.tight_layout()

            return h1, h2, h3

    def __draw_intensities__(self,
                             logarithm,
                             normalize,
                             cut_value,
                             color_intensity=CONF_DRAWING['color_intensity']):
        """internal funcion: draws phase X,Y, Z.

        Parameters:
            logarithm (bool): If True, intensity is scaled in logarithm
            normalize (bool): If True, max(intensity)=1
            cut_value (float): If not None, cuts the maximum intensity to this value
        """

        Ex_r = reduce_matrix_size(self.reduce_matrix, self.x, self.y, self.Ex)
        Ey_r = reduce_matrix_size(self.reduce_matrix, self.x, self.y, self.Ey)
        Ez_r = reduce_matrix_size(self.reduce_matrix, self.x, self.y, self.Ez)
        tx, ty = rcParams['figure.figsize']

        intensity1 = np.abs(Ex_r)**2
        intensity1 = normalize_draw(intensity1, logarithm, normalize,
                                    cut_value)

        intensity2 = np.abs(Ey_r)**2
        intensity2 = normalize_draw(intensity2, logarithm, normalize,
                                    cut_value)

        intensity3 = np.abs(Ez_r)**2
        intensity3 = normalize_draw(intensity3, logarithm, normalize,
                                    cut_value)

        intensity_max = np.max(
            (intensity1.max(), intensity2.max(), intensity3.max()))

        if intensity3.max() < 1e-15:

            plt.figure(figsize=(2 * tx, ty))

            h1 = plt.subplot(1, 2, 1)

            __draw1__(self, intensity1, color_intensity, "$I_x$")
            plt.clim(0, intensity_max)

            h2 = plt.subplot(1, 2, 2)

            __draw1__(self, intensity2, color_intensity, "$I_y$")
            plt.clim(0, intensity_max)

            plt.subplots_adjust(left=0,
                                bottom=0,
                                right=1,
                                top=1,
                                wspace=0.05,
                                hspace=0)
            plt.tight_layout()

            return h1, h2
        else:

            plt.figure(figsize=(3 * tx, ty))

            h1 = plt.subplot(1, 3, 1)

            __draw1__(self, intensity1, color_intensity, "$I_x$")
            # plt.clim(0, intensity_max)

            h2 = plt.subplot(1, 3, 2)

            __draw1__(self, intensity2, color_intensity, "$I_y$")
            # plt.clim(0, intensity_max)

            h3 = plt.subplot(1, 3, 3)

            __draw1__(self, intensity3, color_intensity, "$I_z$")
            # plt.clim(0, intensity_max)

            plt.subplots_adjust(left=0,
                                bottom=0,
                                right=1,
                                top=1,
                                wspace=0.05,
                                hspace=0)
            plt.tight_layout()

            return h1, h2, h3

    def __draw_intensities_rz__(
            self,
            logarithm,
            normalize,
            cut_value,
            color_intensity=CONF_DRAWING['color_intensity']):
        """internal funcion: draws intensity X,Y.

        Parameters:
            logarithm (bool): If True, intensity is scaled in logarithm
            normalize (bool): If True, max(intensity)=1
            cut_value (float): If not None, cuts the maximum intensity to this value
        """

        Ex_r = reduce_matrix_size(self.reduce_matrix, self.x, self.y, self.Ex)
        Ey_r = reduce_matrix_size(self.reduce_matrix, self.x, self.y, self.Ey)
        Ez_r = reduce_matrix_size(self.reduce_matrix, self.x, self.y, self.Ez)
        tx, ty = rcParams['figure.figsize']

        intensity_r = np.abs(Ex_r)**2 + np.abs(Ey_r)**2
        intensity_r = normalize_draw(intensity_r, logarithm, normalize,
                                     cut_value)

        intensity_z = np.abs(Ez_r)**2
        intensity_z = normalize_draw(intensity_z, logarithm, normalize,
                                     cut_value)

        intensity_max = np.max((intensity_r, intensity_z))

        plt.figure(figsize=(2 * tx, ty))

        h1 = plt.subplot(1, 2, 1)

        __draw1__(self, intensity_r, color_intensity, "$I_r$")
        plt.clim(0, intensity_max)

        h2 = plt.subplot(1, 2, 2)

        __draw1__(self, intensity_z, color_intensity, "$I_z$")

        plt.subplots_adjust(left=0,
                            bottom=0,
                            right=1,
                            top=1,
                            wspace=0.05,
                            hspace=0)
        plt.tight_layout()

        return h1, h2

    def __draw_fields__(self,
                        logarithm,
                        normalize,
                        cut_value,
                        color_intensity=CONF_DRAWING['color_intensity'],
                        color_phase=CONF_DRAWING['color_phase']):
        """__internal__: draws amplitude and phase in 2x2 drawing

        Parameters:
            logarithm (bool): If True, intensity is scaled in logarithm
            normalize (bool): If True, max(intensity)=1
            title (str): title of figure
            cut_value (float): If not None, cuts the maximum intensity to this value

        """

        Ex_r = reduce_matrix_size(self.reduce_matrix, self.x, self.y, self.Ex)
        Ey_r = reduce_matrix_size(self.reduce_matrix, self.x, self.y, self.Ey)

        intensity_x = np.abs(Ex_r)**2
        intensity_x = normalize_draw(intensity_x, logarithm, normalize,
                                     cut_value)

        intensity_y = np.abs(Ey_r)**2
        intensity_y = normalize_draw(intensity_y, logarithm, normalize,
                                     cut_value)

        intensity_max = np.max((intensity_x.max(), intensity_y.max()))
        tx, ty = rcParams['figure.figsize']

        plt.figure(figsize=(2 * tx, 2 * ty))

        h1 = plt.subplot(2, 2, 1)

        __draw1__(self, intensity_x, color_intensity, "$I_x$")
        plt.clim(0, intensity_max)

        h2 = plt.subplot(2, 2, 2)
        __draw1__(self, intensity_y, color_intensity, "$I_y$")
        plt.clim(0, intensity_max)

        h3 = plt.subplot(2, 2, 3)
        phase = np.angle(self.Ex)
        phase[intensity_x < percentage_intensity * (intensity_x.max())] = 0

        __draw1__(self, phase / degrees, color_phase, "$\phi_x$")
        plt.clim(-180, 180)

        h4 = plt.subplot(2, 2, 4)
        phase = np.angle(self.Ey)
        phase[intensity_y < percentage_intensity * (intensity_y.max())] = 0

        __draw1__(self, phase / degrees, color_phase, "$\phi_y$")
        plt.clim(-180, 180)
        h4 = plt.gca()
        plt.subplots_adjust(left=0,
                            bottom=0,
                            right=1,
                            top=1,
                            wspace=0.05,
                            hspace=0)
        plt.tight_layout()
        return h1, h2, h3, h4

    def __draw_stokes__(self,
                        logarithm,
                        normalize,
                        cut_value,
                        color_intensity=CONF_DRAWING['color_intensity'],
                        color_stokes=CONF_DRAWING['color_stokes']):
        """__internal__: computes and draws CI, CQ, CU, CV parameters
        """

        tx, ty = rcParams['figure.figsize']

        S0, S1, S2, S3 = self.polarization_states(matrix=True)
        S0 = normalize_draw(S0, logarithm, normalize, cut_value)
        S1 = normalize_draw(S1, logarithm, normalize, cut_value)
        S2 = normalize_draw(S2, logarithm, normalize, cut_value)
        S3 = normalize_draw(S3, logarithm, normalize, cut_value)

        intensity_max = S0.max()

        plt.figure(figsize=(2 * tx, 2 * ty))
        h1 = plt.subplot(2, 2, 1)
        __draw1__(self, S0, color_intensity, "$S_0$")
        plt.clim(0, intensity_max)

        h2 = plt.subplot(2, 2, 2)
        __draw1__(self, S1, color_stokes, "$S_1$")
        plt.clim(-intensity_max, intensity_max)

        h3 = plt.subplot(2, 2, 3)
        __draw1__(self, S2, color_stokes, "$S_2$")
        plt.clim(-intensity_max, intensity_max)

        h4 = plt.subplot(2, 2, 4)
        __draw1__(self, S3, color_stokes, "$S_3$")
        plt.clim(-intensity_max, intensity_max)

        plt.subplots_adjust(left=0,
                            bottom=0,
                            right=1,
                            top=1,
                            wspace=0.05,
                            hspace=0)
        plt.tight_layout()
        return (h1, h2, h3, h4)

    def __draw_param_ellipse__(self,
                               color_intensity=CONF_DRAWING['color_intensity'],
                               color_phase=CONF_DRAWING['color_phase']):
        """__internal__: computes and draws polariations ellipses
        """
        A, B, theta, h = self.polarization_ellipse(pol_state=None, matrix=True)

        A = reduce_matrix_size(self.reduce_matrix, self.x, self.y, A)
        B = reduce_matrix_size(self.reduce_matrix, self.x, self.y, B)
        theta = reduce_matrix_size(self.reduce_matrix, self.x, self.y, theta)
        h = reduce_matrix_size(self.reduce_matrix, self.x, self.y, h)

        tx, ty = rcParams['figure.figsize']

        plt.figure(figsize=(2 * tx, 2 * ty))

        max_intensity = max(A.max(), B.max())

        h1 = plt.subplot(2, 2, 1)
        __draw1__(self, A, color_intensity, "$A$")
        plt.clim(0, max_intensity)
        h2 = plt.subplot(2, 2, 2)
        __draw1__(self, B, color_intensity, "$B$")
        plt.clim(0, max_intensity)

        h3 = plt.subplot(2, 2, 3)
        __draw1__(self, theta / degrees, color_phase, "$\phi$")
        plt.clim(-180, 180)
        h4 = plt.subplot(2, 2, 4)
        __draw1__(self, h, "gist_heat", "$h$")
        plt.tight_layout()
        return (h1, h2, h3, h4)

    def __draw_ellipses__(self,
                          logarithm=False,
                          normalize=False,
                          cut_value='',
                          num_ellipses=(21, 21),
                          amplification=0.75,
                          color_line='w',
                          line_width=1,
                          draw_arrow=True,
                          head_width=2,
                          ax=False,
                          color_intensity=CONF_DRAWING['color_intensity']):
        """__internal__: draw ellipses

        Parameters:
            num_ellipses (int): number of ellipses for parameters_ellipse
        """

        percentage_intensity = CONF_DRAWING['percentage_intensity']
        intensity_max = (np.abs(self.Ex)**2 + np.abs(self.Ey)**2).max()

        Dx = self.x[-1] - self.x[0]
        Dy = self.y[-1] - self.y[0]
        size_x = Dx / (num_ellipses[0])
        size_y = Dy / (num_ellipses[1])
        x_centers = size_x / 2 + size_x * np.array(range(0, num_ellipses[0]))
        y_centers = size_y / 2 + size_y * np.array(range(0, num_ellipses[1]))

        num_x, num_y = len(self.x), len(self.y)
        ix_centers = num_x / (num_ellipses[0])
        iy_centers = num_y / (num_ellipses[1])

        ix_centers = (np.round(
            ix_centers / 2 +
            ix_centers * np.array(range(0, num_ellipses[0])))).astype('int')
        iy_centers = (np.round(
            iy_centers / 2 +
            iy_centers * np.array(range(0, num_ellipses[1])))).astype('int')

        Ix_centers, Iy_centers = np.meshgrid(ix_centers.astype('int'),
                                             iy_centers.astype('int'))

        verbose = False
        if verbose is True:
            print(num_x, num_y, ix_centers, iy_centers)
            print(Dx, Dy, size_x, size_y)
            print(x_centers, y_centers)
            print(Ix_centers, Iy_centers)

        E0x = self.Ex[Iy_centers, Ix_centers]
        E0y = self.Ey[Iy_centers, Ix_centers]

        angles = np.linspace(0, 360 * degrees, 64)

        if ax is False:
            self.draw('intensity',
                      logarithm=logarithm,
                      color_intensity=color_intensity)
            ax = plt.gca()

        for i, xi in enumerate(ix_centers):
            for j, yj in enumerate(iy_centers):
                Ex = np.real(E0x[j, i] * np.exp(1j * angles))
                Ey = np.real(E0y[j, i] * np.exp(1j * angles))

                max_r = np.sqrt(np.abs(Ex)**2 + np.abs(Ey)**2).max()
                size_dim = min(size_x, size_y)

                if max_r > 0 and max_r**2 > percentage_intensity * intensity_max:

                    Ex = Ex / max_r * size_dim * amplification / 2 + (
                        +self.x[int(xi)])
                    Ey = Ey / max_r * size_dim * amplification / 2 + self.y[
                        int(yj)]

                    ax.plot(Ex, Ey, color_line, lw=line_width)
                    if draw_arrow:
                        ax.arrow(Ex[0],
                                 Ey[0],
                                 Ex[0] - Ex[1],
                                 Ey[0] - Ey[1],
                                 width=0,
                                 head_width=head_width,
                                 fc=color_line,
                                 ec=color_line,
                                 length_includes_head=False)
                # else:
                #     print(max_r, intensity_max,
                #           percentage_intensity * intensity_max)


def __draw1__(hdl, image, colormap, title='', has_max=False, only_image=False):
    """Draws image

    Parameters:
        image (numpy.array): array with drawing
        colormap (str): colormap
        title (str): title of drawing
    """
    extension = [hdl.x[0], hdl.x[-1], hdl.y[0], hdl.y[-1]]

    h = plt.imshow(image,
                   interpolation='bilinear',
                   aspect='auto',
                   origin='lower',
                   extent=extension)
    h.set_cmap(colormap)
    plt.axis('scaled')
    plt.axis(extension)

    if only_image is True:
        plt.axis('off')
        return h

    plt.title(title, fontsize=16)

    if has_max is True:
        text_up = "{}".format(image.max())
        plt.text(hdl.x.max(),
                 hdl.y.max(),
                 text_up,
                 fontsize=14,
                 bbox=dict(edgecolor='white', facecolor='white', alpha=0.75))

        text_down = "{}".format(image.min())
        plt.text(hdl.x.max(),
                 hdl.y.min(),
                 text_down,
                 fontsize=14,
                 bbox=dict(edgecolor='white', facecolor='white', alpha=0.75))

    plt.xlabel("$x  (\mu m)$")
    plt.ylabel("$y  (\mu m)$")
    if colormap is not None:
        plt.colorbar(orientation='horizontal', shrink=0.66)
        h.set_clim(0, image.max())

    return h


def _compute1Elipse__(x0, y0, A, B, theta, h=0, amplification=1):
    """computes polarization ellipse for drawing

    Parameters:
        x0 (float): position x of ellipse
        y0 (float): position y of ellipse
        A (float): axis 1 of ellipse
        B (float): axis 2 of ellipse
        theta (float): angle of ellipse
        h (float): to remove
        amplification (float): increase of size of ellipse

    TODO:
        remove hs
    """
    # esto es para verlo más grande
    A = A * amplification
    B = B * amplification

    fi = np.linspace(0, 2 * np.pi, 64)
    cf = np.cos(fi - theta)
    sf = np.sin(fi - theta)

    r = 1 / np.sqrt(np.abs(cf / (A + eps)**2 + sf**2 / (B + eps)**2))

    x = r * np.cos(fi) + x0
    y = r * np.sin(fi) + y0

    return x, y


def polarization_ellipse(self, pol_state=None, matrix=False):
    """returns A, B, theta, h polarization parameter of elipses

    Parameters:
        pol_state (None or (I, Q, U, V) ): Polarization state previously computed
        Matrix (bool): if True returns Matrix, else Scalar_field_XY

    Returns:
        A, B, theta, h for Matrix=True
        CA, CB, Ctheta, Ch for Matrix=False
    """
    if pol_state is None:
        I, Q, U, V = self.polarization_states(matrix=True)
    else:
        I, Q, U, V = pol_state
        I = I.u
        Q = Q.u
        U = U.u
        V = V.u

    Ip = np.sqrt(Q**2 + U**2 + V**2)
    L = Q + 1.j * U

    A = np.real(np.sqrt(0.5 * (Ip + np.abs(L))))
    B = np.real(np.sqrt(0.5 * (Ip - np.abs(L))))
    theta = 0.5 * np.angle(L)
    h = np.sign(V)

    if matrix is True:
        return A, B, theta, h
    else:
        CA = Scalar_field_XY(x=self.x, y=self.y, wavelength=self.wavelength)
        CB = Scalar_field_XY(x=self.x, y=self.y, wavelength=self.wavelength)
        Ctheta = Scalar_field_XY(x=self.x,
                                 y=self.y,
                                 wavelength=self.wavelength)
        Ch = Scalar_field_XY(x=self.x, y=self.y, wavelength=self.wavelength)

        CA.u = A
        CB.u = B
        Ctheta.u = theta
        Ch.u = h
        return (CA, CB, Ctheta, Ch)

        I = I.u
        Q = Q.u
        U = U.u
        V = V.u

    Ip = np.sqrt(Q**2 + U**2 + V**2)
    L = Q + 1.j * U

    A = np.real(np.sqrt(0.5 * (Ip + np.abs(L))))
    B = np.real(np.sqrt(0.5 * (Ip - np.abs(L))))
    theta = 0.5 * np.angle(L)
    h = np.sign(V)

    if matrix is True:
        return A, B, theta, h
    else:
        CA = Scalar_field_XY(x=self.x, y=self.y, wavelength=self.wavelength)
        CB = Scalar_field_XY(x=self.x, y=self.y, wavelength=self.wavelength)
        Ctheta = Scalar_field_XY(x=self.x,
                                 y=self.y,
                                 wavelength=self.wavelength)
        Ch = Scalar_field_XY(x=self.x, y=self.y, wavelength=self.wavelength)

        CA.u = A
        CB.u = B
        Ctheta.u = theta
        Ch.u = h
        return (CA, CB, Ctheta, Ch)
