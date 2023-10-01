import numpy as np

# ---General Variables----

FILEPATH = '/home/bego/Gaia_DR3/DR3/'
FILENAME = 'DR3_photogeo_ruwe14_GAL6D.fits'

VPHI_SUN = -236  #Azimutal velocity of the Sun in km/s


R_CUT = 8.5 #galactocentric R of the phase spiral we want to fit in kpc
R_BIN = 0.5 #R bin in kpc
PHI_CUT = 0 #galactocentric phi
PHI_BIN = 10 #phi bin


WRAPS = 3 #Estimated wraps of the phase spiral

RZ_MIN = 0.2   #Min and max polar rho for unwrapping the spiral
RZ_MAX = 1

PHIZ_BIN = 0.01

#RANSAC settings
RANSAC_IT = 500
RANSAC_THRESHOLD = 0.5
# Wavelet params
RZ_BIN = 0.005
BLEND = 16
WAVELET_SIZE = 8  # Pixels of rz_bins to detect

#HDBSCAN settings
MIN_CLUSTER_SIZE = 20
MIN_SAMPLES = 5

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


#-------MCMC config------
DRAWs = 2000
TUNE = 1000
CORES = 2