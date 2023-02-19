import numpy as np

#---General Variables----
Vphi_sun = -236
R_bin = 0.5
phi_bin = 10
phi_cut = 0
wraps = 3
rz_min = 0.2
rz_max = 1
phiz_bin = 0.01
ransac_it = 500
ransac_threshold = 0.5
# Wavelet params
rz_bin = 0.005
blend = 16
wavelet_size = 8  # Pixels of rz_bins to detect


#-------- For 2D histogram----
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