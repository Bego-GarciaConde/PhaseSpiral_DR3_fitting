# authors: Begoña García-Conde and Marcel Bernet

from astropy.table import Table
from GaiaVolume import *
from config import *


def main():
    print("Reading data...")
    dat = Table.read(FILEPATH + FILENAME, format='fits')
    df = dat.to_pandas()
    # We select the volume of the Gaia data
    volume = GaiaVolume(data=df, r_cut=R_CUT, phi_cut=PHI_CUT)
    volume.apply_unwrap()
    volume.apply_ransac_filter()
    volume.min_squared_spiral_fittin()
    volume.mcmc_spiral_fitting()


if __name__ == "__main__":
    main()
