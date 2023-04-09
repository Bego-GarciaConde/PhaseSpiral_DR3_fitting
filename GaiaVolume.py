import hdbscan
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from skimage import feature
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

from utils import *


class GaiaVolume():
    def __init__(self, data, r_cut, phi_cut=phi_cut, r_bin=0.8, phi_bin=phi_bin):
        self.r_cut = r_cut
        self.phi_cut = phi_cut
        self.df = None
        self.scales = None
        self.H = None
        self.xedges = None
        self.yedges = None

        self.data_polar = None

        self.unwrapped_phiz = None
        self.unwrapped_rhoz = None

        def filter_volume():
            df = data[(np.abs(data['phi'] - phi_cut) < phi_bin) &
                      (np.abs(data['R'] * data['Vphi'] - r_cut * Vphi_sun) < np.abs(r_bin * Vphi_sun))]
            return df

        self.df = filter_volume()

    def find_scale(self):
        """
        Scale Vz and Z, so the ranges of the spiral is equivalent
        """
        z_scale = np.std(self.df["Z"])
        vz_scale = np.std(self.df["VZ"])
        normalization = np.std(self.df["VZ"]) / np.std(self.df["Z"])
        print(normalization)
        r_pre = np.sqrt((self.df["VZ"] / vz_scale) ** 2 + (self.df["Z"] / z_scale) ** 2)
        phi_pre = np.arctan2(self.df["VZ"] / vz_scale, self.df["Z"] / z_scale)
        p_limit = np.percentile(r_pre, 85)
        self.scales = [z_scale, vz_scale, p_limit]
        return z_scale, vz_scale, p_limit

    def apply_edge_detector(self):
        """
        The edge detector finds the "edges" of the spiral
        """
        H, xedges, yedges = calc_histogram_cartesian(self.df, z_bins, vz_bins, weights=None)
        self.H = H
        self.xedges = xedges
        self.yedges = yedges

        m = feature.canny(np.log10(self.H), sigma=3)
        z_array = []
        vz_array = []

        for i, z_i in enumerate(zcentres):
            for j, vz_i in enumerate(vzcentres):
                if m[i, j]:
                    z_array.append(z_i)
                    vz_array.append(vz_i)
        return np.array(z_array), np.array(vz_array)

    def apply_unwrap(self):
        """
        Applies an edge detector and transforms cartesian coordinates into polar
        These polar coordinates are wrapped
        """
        z_array, vz_array = self.apply_edge_detector()
        z_scale, vz_scale, p_limit = self.find_scale()
        print(z_scale, vz_scale, p_limit)
        ind = (np.sqrt((vz_array / vz_scale) ** 2 + (z_array / z_scale) ** 2) < p_limit)
        z_filt = z_array[ind]
        vz_filt = vz_array[ind]

        rz = np.sqrt((vz_filt / vz_scale) ** 2 + (z_filt / z_scale) ** 2)
        phiz = np.arctan2(vz_filt / vz_scale, z_filt / z_scale)

        data = {'Phi': np.array(phiz), 'R': np.array(rz)}
        self.data_polar = pd.DataFrame(data)

    def apply_ransac_filter(self):

        # The first approach is an arquimedean spiral: straght line in polar coordinates
        # In this space there could be several wraps, we colate them and sort them to have a continous line

        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
        clusterer.fit(self.data_polar)
        labels = clusterer.labels_
        unique_labels = np.unique(labels)

        # Find the coordinates of each cluster
        central_coords = np.zeros((len(unique_labels), 2))

        for i, label in enumerate(unique_labels):
            if label == -1:  # Skip the noise points
                continue

            cluster_points = self.data_polar[labels == label]
            central_coords[i] = np.median(cluster_points, axis=0)

        sorted_labels = central_coords[:, 1].argsort()

        self.data_polar["Phi_trans"] = self.data_polar["Phi"].copy()
        count = 0
        # Sort them from lower r (inner spiral) to higher (outer spiral)
        for i, label in enumerate(sorted_labels):
            print(label)
            cluster_points = self.data_polar[labels == label - 1]

            if i != 0:
                print(f"Previous: {np.mean(cluster_points_prev['Phi'])} Now: {np.median(cluster_points['Phi'])} ")
                if np.median(cluster_points_prev["Phi"]) > np.median(cluster_points["Phi"]):
                    count += 1
            # It can detect several groups in the same wrap
            self.data_polar.loc[cluster_points.index, "Phi_trans"] = cluster_points["Phi"] + count * 2 * np.pi
            #        print(f"for label {label} situated at r {central_coords[label, 1]}, multiple is {count}")
            cluster_points_prev = cluster_points

        phiz = np.array(self.data_polar["Phi_trans"]).reshape(-1, 1)
        rz = np.array(self.data_polar["R"]).reshape(-1, 1)

        lr = linear_model.LinearRegression()
        lr.fit(phiz, rz)

        ransac = linear_model.RANSACRegressor(max_trials=10000)
        ransac.fit(phiz, rz)
        inlier_mask = ransac.inlier_mask_
        outlier_mask = np.logical_not(inlier_mask)
        line_X = np.arange(phiz.min(), phiz.max())[:, np.newaxis]
        line_y = lr.predict(line_X)
        line_y_ransac = ransac.predict(line_X)

        self.unwrapped_phiz = phiz[inlier_mask]
        self.unwrapped_rhoz = rz[inlier_mask]

    def spiral_fitting(self):
        # ----- LINEAR, QUADRATIC, LOG REGRESSION---- WHICH ONE FITS BETTER?----
        regr = LinearRegression()

        quadratic = PolynomialFeatures(degree=2)
        cubic = PolynomialFeatures(degree=3)

        Phi_quad = quadratic.fit_transform(self.unwrapped_phiz)
        Phi_cubic = cubic.fit_transform(self.unwrapped_phiz)

        Phi_fit = np.arange(self.unwrapped_phiz.min(), self.unwrapped_phiz.max() + 1, 0.1)[:, np.newaxis]

        regr = regr.fit(self.unwrapped_phiz, self.unwrapped_rhoz)
        r_lin_fit = regr.predict(Phi_fit)
        linear_r2 = r2_score(self.unwrapped_rhoz, regr.predict(self.unwrapped_phiz))
        # print(f"reg coef is {regr.coef_}")

        regr = regr.fit(self.unwrapped_phiz, np.log(self.unwrapped_rhoz))
        r_log_fit = regr.predict(Phi_fit)
        log_r2 = r2_score(np.log(self.unwrapped_rhoz), regr.predict(self.unwrapped_phiz))

        regr = regr.fit(Phi_quad, self.unwrapped_rhoz)
        r_quad_fit = regr.predict(quadratic.fit_transform(Phi_fit))
        quadratic_r2 = r2_score(self.unwrapped_rhoz, regr.predict(Phi_quad))

        regr = regr.fit(Phi_cubic, self.unwrapped_rhoz)
        r_cubic_fit = regr.predict(cubic.fit_transform(Phi_fit))
        cubic_r2 = r2_score(self.unwrapped_rhoz, regr.predict(Phi_cubic))

        # -----PLOT RESULTS-----
        # TODO: fix axis labels
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        sc = ax[0].imshow(self.H.T, origin='lower', cmap='inferno_r',
                          extent=[self.xedges[0], self.xedges[-1], self.yedges[0], self.yedges[-1]],
                          aspect='auto')

        ax[1].scatter(self.unwrapped_phiz, self.unwrapped_rhoz, label='Unwrapped spiral', color='blue', alpha=0.3)

        ax[1].plot(Phi_fit, r_lin_fit, label='linear (d=1), $R^2=%.2f$' % linear_r2, color='blue',
                   lw=2, linestyle=':', alpha=0.5)

        ax[1].plot(Phi_fit, r_quad_fit, label='quadratic (d=2), $R^2=%.2f$' % quadratic_r2,
                   color='red', lw=2, linestyle='-', alpha=0.5)

        ax[1].plot(Phi_fit, r_cubic_fit, label='cubic (d=3), $R^2=%.2f$' % cubic_r2,
                   color='green', lw=2, linestyle='--', alpha=0.5)

        ax[1].plot(Phi_fit, np.exp(r_log_fit),
                   label='Logarithmic, $R^2=%.2f$' % log_r2, color='purple', lw=2, linestyle='--', alpha=0.5)

        ax[1].legend(loc='upper left')

        ax[0].set_box_aspect(1)
        ax[1].set_box_aspect(1)

        ax[0].set_title(r"$R_{gal}$=" + f"{r_cut}" + r", $\Phi_{gal}$=" + f"{phi_cut}")

        ax[0].set_xlabel("Z [kpc]")
        ax[0].set_ylabel("$V_{Z}$ [kpc]")

        ax[1].set_xlabel("$\Phi_{Z}$")
        ax[1].set_ylabel("$R_{Z}$")
        plt.savefig(f"results/phase_spiral_R{r_cut}_phi{phi_cut}.png", bbox_inches="tight")
        plt.show()
