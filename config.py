import numpy as np

# ---General Variables----

filepath = '/home/bego/Gaia_DR3/DR3/'
filename = 'DR3_photogeo_ruwe14_GAL6D.fits'

Vphi_sun = -236  #Azimutal velocity of the Sun in km/s


r_cut = 9 #galactocentric R of the phase spiral we want to fit in kpc
R_bin = 0.5 #R bin in kpc
phi_cut = -2 #galactocentric phi
phi_bin = 10 #phi bin


wraps = 3 #Estimated wraps of the phase spiral

rz_min = 0.2   #Min and max polar rho for unwrapping the spiral
rz_max = 1

phiz_bin = 0.01

#RANSAC settings
ransac_it = 500
ransac_threshold = 0.5
# Wavelet params
rz_bin = 0.005
blend = 16
wavelet_size = 8  # Pixels of rz_bins to detect

#HDBSCAN settings
min_cluster_size = 20
min_samples = 5

# -------- For 2D histogram----
z_bins = np.arange(-1, 1, 0.01)
vz_bins = np.arange(-60, 60, 1)
# print(len(z_bins),len(vz_bins))
zcentres = 0.5 * (z_bins[1:] + z_bins[:-1])
vzcentres = 0.5 * (vz_bins[1:] + vz_bins[:-1])

rangex = [-1, 1.01]
rangey = [-60, 60]
binsx = (np.max(z_bins) - np.min(z_bins)) / len(z_bins)
binsy = (np.max(vz_bins) - np.min(vz_bins)) / len(vz_bins)
aspect = (rangex[1] - rangex[0]) / (rangey[1] - rangey[0])
deltax = (rangex[1] - rangex[0]) / (binsx * 1.)
deltay = (rangey[1] - rangey[0]) / (binsy * 1.)
