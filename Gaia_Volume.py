
from variables_config import *
from function_tools import *
import numpy as np
import pandas as pd
import cv2
from skimage import feature

class Gaia_Volume():
    def __init__(self, data, r_cut, phi_cut=0, r_bin=0.8, phi_bin=3):
        self.r_cut = r_cut
        self.phi_cut = phi_cut
        self.df = None
        self.phi = None
        self.rho = None
        self.phi = None
        self.H = None
        self.xedges = None
        self.yedges = None

        def filter_volume():
            df = data[(np.abs(data['phi'] - phi_cut) < phi_bin) &
                      (np.abs(data['R'] * data['Vphi'] - r_cut * Vphi_sun) < np.abs(r_bin * Vphi_sun))]
            return df

        self.df = filter_volume()

    def find_scale(self):
        z_scale = np.std(self.df["Z"])
        vz_scale = np.std(self.df["VZ"])
        normalization = np.std(self.df["VZ"]) / np.std(self.df["Z"])
        print(normalization)
        r_pre = np.sqrt((self.df["VZ"] / vz_scale) ** 2 + (self.df["Z"] / z_scale) ** 2)
        phi_pre = np.arctan2(self.df["VZ"] / vz_scale, self.df["Z"] / z_scale)
        p_limit = np.percentile(r_pre, 85)
        return z_scale, vz_scale, p_limit

    def apply_edge_detector(self):
        H, xedges, yedges = calc_histogram_cartesian(self.df, z_bins, vz_bins, weights=None)
        self.H = H
        self.xedges = xedges
        self.yedges = yedges

        m = feature.canny(np.log10(self.H), sigma=3)
        z_array = []
        vz_array = []
        # zcentres = 0.5 * (self.xedges[1:] + self.xedges[:-1])
        # vzcentres = 0.5 * (self.yedges[1:] + self.yedges[:-1])
        for i, z_i in enumerate(zcentres):
            for j, vz_i in enumerate(vzcentres):
                if m[i, j]:
                    z_array.append(z_i)
                    vz_array.append(vz_i)
        return np.array(z_array), np.array(vz_array)

    def apply_unwrap(self):
        z_array, vz_array = self.apply_edge_detector()
        z_scale, vz_scale, p_limit = self.find_scale()
        print(z_scale, vz_scale, p_limit)
        ind = (np.sqrt((vz_array / vz_scale) ** 2 + (z_array / z_scale) ** 2) < p_limit)
        z_filt = z_array[ind]
        vz_filt = vz_array[ind]

        rz = np.sqrt((vz_filt / vz_scale) ** 2 + (z_filt / z_scale) ** 2)
        phiz = np.arctan2(vz_filt / vz_scale, z_filt / z_scale)

        return z_filt, vz_filt, rz, phiz
    #   phi_re, r_re = unwrap(rz, phiz, wraps = wraps)
