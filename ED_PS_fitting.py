# authors: Begoña García-Conde and Marcel Bernet


from os.path import expanduser

home = expanduser("~")
# from skimage import feature
from sklearn import linear_model, datasets
from sklearn.linear_model import RANSACRegressor
from sklearn.datasets import make_regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression, RANSACRegressor
import cv2
from astropy.table import Table
from function_tools import *



from Gaia_Volume import *
import numpy as np
import pandas as pd


from sklearn.preprocessing import PolynomialFeatures

from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression, RANSACRegressor

from astropy.table import Table
import hdbscan

rcParams['font.family'] = 'sans-serif'
rcParams["font.size"] = 8
rcParams["font.family"] = "Arial"
rcParams['mathtext.fontset'] = 'custom'
rcParams['mathtext.it'] = 'Arial:italic'
rcParams['mathtext.rm'] = 'Arial'
rcParams['pdf.fonttype'] = 42


def main():
    print("Reading data...")
    dat = Table.read('/home/bego/Gaia_DR3/DR3/DR3_photogeo_ruwe14_GAL6D.fits', format='fits')
    df = dat.to_pandas()

    # We select the volume of the Gaia data
    volume = Gaia_Volume(data=df, r_cut=8.5)

    # Transform Z, Vz cartesian space into rz, phiz (polar)
    z_filt, vz_filt, rz, phiz = volume.apply_unwrap()
    z_array, vz_array = volume.apply_edge_detector()

    data = {'Phi': np.array(phiz), 'R': np.array(rz)}

    # The first approach is an arquimedean spiral: straght line in polar coordinates
    # In this space there could be several wraps, we colate them and sort them to have a continous line
    spiral = pd.DataFrame(data)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=20, min_samples=5)
    clusterer.fit(spiral)
    labels = clusterer.labels_
    unique_labels = np.unique(labels)

    # Find the coordinates of each cluster
    central_coords = np.zeros((len(unique_labels), 2))

    for i, label in enumerate(unique_labels):
        if label == -1:  # Skip the noise points
            continue

        cluster_points = spiral[labels == label]
        central_coords[i] = np.median(cluster_points, axis=0)

    sorted_labels = central_coords[:, 1].argsort()

    spiral["Phi_trans"] = spiral["Phi"].copy()
    count = 0
    # Sort them from lower r (inner spiral) to higher (outer spiral)
    for i, label in enumerate(sorted_labels):
        print(label)
        cluster_points = spiral[labels == label - 1]

        if i != 0:
            print(f"Previous: {np.mean(cluster_points_prev['Phi'])} Now: {np.median(cluster_points['Phi'])} ")
            if np.median(cluster_points_prev["Phi"]) > np.median(cluster_points["Phi"]):
                count += 1
        # It can detect several groups in the same wrap
        spiral.loc[cluster_points.index, "Phi_trans"] = cluster_points["Phi"] + count * 2 * np.pi
        #        print(f"for label {label} situated at r {central_coords[label, 1]}, multiple is {count}")
        cluster_points_prev = cluster_points

    # We apply a RANSAC algorithm to clean the data from outliers and do a first apprach to archimedean spiral
    # ----- RANSAC ALGORITHM TO DISCARD OUTLIERS --------

    # TODO: Is there another way to filter inliers/outliers only with hdbscan?
    phiz = np.array(spiral["Phi_trans"]).reshape(-1, 1)
    rz = np.array(spiral["R"]).reshape(-1, 1)
    lr = linear_model.LinearRegression()
    lr.fit(phiz, rz)

    ransac = linear_model.RANSACRegressor(max_trials=10000)
    ransac.fit(phiz, rz)
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)
    line_X = np.arange(phiz.min(), phiz.max())[:, np.newaxis]
    line_y = lr.predict(line_X)
    line_y_ransac = ransac.predict(line_X)

    phiz = phiz[inlier_mask]
    rz = rz[inlier_mask]
    # ----- LINEAR, QUADRATIC, LOG REGRESSION---- WHICH ONE FITS BETTER?----
    regr = LinearRegression()
    quadratic = PolynomialFeatures(degree=2)
    cubic = PolynomialFeatures(degree=3)
    Phi_quad = quadratic.fit_transform(phiz)
    Phi_cubic = cubic.fit_transform(phiz)

    Phi_fit = np.arange(phiz.min(), phiz.max() + 1, 0.1)[:, np.newaxis]

    regr = regr.fit(phiz, rz)
    r_lin_fit = regr.predict(Phi_fit)
    linear_r2 = r2_score(rz, regr.predict(phiz))
    print(f"reg coef is {regr.coef_}")

    regr = regr.fit(phiz, np.log(rz))
    r_log_fit = regr.predict(Phi_fit)
    log_r2 = r2_score(np.log(rz), regr.predict(phiz))

    regr = regr.fit(Phi_quad, rz)
    r_quad_fit = regr.predict(quadratic.fit_transform(Phi_fit))
    quadratic_r2 = r2_score(rz, regr.predict(Phi_quad))

    regr = regr.fit(Phi_cubic, rz)
    r_cubic_fit = regr.predict(cubic.fit_transform(Phi_fit))
    cubic_r2 = r2_score(rz, regr.predict(Phi_cubic))

    # -----PLOT RESULTS-----
    # TODO: fix axis labels
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    sc = ax[0].imshow(volume.H.T, origin='lower', cmap='inferno_r',
                      extent=[volume.xedges[0], volume.xedges[-1], volume.yedges[0], volume.yedges[-1]],
                      aspect='auto')

    ax[1].scatter(phiz, rz, label='Unwrapped spiral', color='blue', alpha=0.3)

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
    plt.show()
    plt.close()





if __name__ == "__main__":
    main()
