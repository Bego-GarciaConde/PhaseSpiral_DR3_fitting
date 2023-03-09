
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.colors as mcolors
import matplotlib.gridspec
from matplotlib import rcParams
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from scipy import stats



from variables_config import *

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged


def fit_curve(a, data):
    theta = (data[:, 0] + np.pi) % (2 * np.pi) - np.pi
    phi = np.arctan2((1 / a[2]) * np.sin(theta), np.cos(theta)) + 2 * np.pi * data[:, 3]
    r_pred = a[0] + a[1] * phi * np.sqrt(np.cos(phi) ** 2 + (a[2] * np.sin(phi)) ** 2)

    return np.linalg.norm(r_pred - data[:, 1])


def calc_histogram_cartesian(df, z_bins, vz_bins, weights=None):
    if weights is None:
        #H,xedges, yedges = np.histogram2d(df['Z'].values, df['VZ'].values,bins=[z_bins,vz_bins],range=[rangex,rangey])
        H, xedges, yedges = np.histogram2d(df['Z'].values, df['VZ'].values, bins=[z_bins, vz_bins],
                                           range=[rangex, rangey],
                                           normed=mcolors.PowerNorm(3))
    elif weights == "Vphi":
        H, xedges, yedges = np.histogram2d(df['Z'].values, df['VZ'].values,
                                           bins=[z_bins, vz_bins], weights=df['Vphi'].values)
        #   H_counts, xedges, yedges = np.histogram2d(df['Z'].values, df['VZ'].values,bins=[z_bins,vz_bins])
        stat = stats.binned_statistic_2d(df['Z'], df['VZ'], df['Vphi'], statistic='mean', bins=[z_bins, vz_bins],
                                         range=[rangex, rangey])
        H = stat[0]
    #   H = np.abs(H/H_counts)
    return H, xedges, yedges



def model_f(x, a, b, h):
    # return a*x + b
    return a * (x - h) ** 2 + b


def model_f_reverse(x, a, b, h):
    # return (x-b)/a
    return np.sqrt((x - b) / a) + h





nbOfColours = 257
base = matplotlib.cm.get_cmap('plasma_r')
mycolorlist = base(np.linspace(0, 1, nbOfColours))
pixelstofade = 2
mycolorlist[0] = [1, 1, 1, 1]
incrementsR = 1. * (1 - mycolorlist[pixelstofade][0]) / pixelstofade
incrementsG = 1. * (1 - mycolorlist[pixelstofade][1]) / pixelstofade
incrementsB = 1. * (1 - mycolorlist[pixelstofade][2]) / pixelstofade
print(incrementsR, incrementsG, incrementsB)
for p in range(pixelstofade):
    n = pixelstofade - p
    mycolorlist[p][0] = mycolorlist[pixelstofade][0] + n * incrementsR
    mycolorlist[p][1] = mycolorlist[pixelstofade][1] + n * incrementsG
    mycolorlist[p][2] = mycolorlist[pixelstofade][2] + n * incrementsB
templatecmap = matplotlib.cm.get_cmap('hot')
mycolormap = templatecmap.from_list('mycustomcolormap', mycolorlist, nbOfColours)
