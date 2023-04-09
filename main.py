# authors: Begoña García-Conde and Marcel Bernet

from astropy.table import Table
from GaiaVolume import *
from config import *


def main():
    print("Reading data...")
    dat = Table.read(filepath + filename, format='fits')
    df = dat.to_pandas()
    # We select the volume of the Gaia data
    volume = GaiaVolume(data=df, r_cut=r_cut)
    volume.apply_unwrap()
    volume.apply_ransac_filter()
    volume.spiral_fitting()


if __name__ == "__main__":
    main()
