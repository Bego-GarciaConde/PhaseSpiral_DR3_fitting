import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.colors as mcolors

import matplotlib.gridspec

from matplotlib import rcParams

from scipy.optimize import minimize
from scipy.interpolate import interp1d

from scipy import stats


from os.path import expanduser

home = expanduser("~")


from skimage import feature
import cv2

from sklearn.linear_model import LinearRegression, RANSACRegressor



rcParams['font.family'] = 'sans-serif'
rcParams["font.size"] = 8
rcParams["font.family"] = "Arial"
rcParams['mathtext.fontset'] = 'custom'
rcParams['mathtext.it'] = 'Arial:italic'
rcParams['mathtext.rm'] = 'Arial'
rcParams['pdf.fonttype']=42

from astropy.table import Table
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score



# build a rectangle in axes coords
left = -0.1
right = 1.1
top = 1.02

left = 0.1
right = 0.95
top = 0.95

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
    if weights == None:
        #   H, xedges, yedges = np.histogram2d(df['Z'].values, df['VZ'].values,bins=[z_bins,vz_bins],range=[rangex,rangey])
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


def filter_data(df, R_cut, phi_cut, R_bin=0.8, phi_bin=3, Vphi_sun=-236):
    #  dfIt = df[(np.abs(df['phi']-phi_cut)<phi_bin) & (df['R']-R_cut<R_bin)]
    dfIt = df[(np.abs(df['phi'] - phi_cut) < phi_bin) & (
                np.abs(df['R'] * df['Vphi'] - R_cut * Vphi_sun) < np.abs(R_bin * Vphi_sun))]
    return dfIt


def model_f(x, a, b, h):
    # return a*x + b
    return a * (x - h) ** 2 + b


def model_f_reverse(x, a, b, h):
    # return (x-b)/a
    return np.sqrt((x - b) / a) + h


def apply_ransac(phi_z, r_z, linear=None):
    if linear == None:
        ransac = RANSACRegressor(base_estimator=LinearRegression(),
                                 min_samples=50, max_trials=100,
                                 loss='absolute_loss', random_state=0,
                                 residual_threshold=0.1)
        #
        # Fit the model
        #
        ransac.fit(phi_z, r_z)
        X_fit = np.arange(phi_z.min(), phi_z.max(), 0.01)[:, np.newaxis]
        line_y_ransac = ransac.predict(X_fit)
        resultado = ransac.score(phi_z, r_z)
        bnds = ((-np.pi, np.pi), (0.01, None), (0.8, 1.2))
    res = minimize(fit_curve, [a_opt_lin[0], a_opt_lin[1], 1], args=(inliers_opt_lin), bounds=bnds)

    a_opt_nl = res.x
    return float(ransac.estimator_.coef_[0]), float(ransac.estimator_.intercept_[0]), X_fit, line_y_ransac



def unwrap(r, phi, wraps=3):
    peakInfo = []
    itPoint = 0

    maxCount = 0
    a_opt = []

    inliers_opt = []
    for i, phi_i in enumerate(phi):
        for phi_n in range(0, wraps):
            peakInfo.append((phi_i + phi_n * 2 * np.pi, r[i], itPoint))
        itPoint += 1

    peakInfo = np.array(peakInfo)
    #  print(peakInfo)
    maxCount = 0
    a_opt = []
    inliers_opt = []
    # Ransac algorithm
    for it in range(ransac_it):
        sample = np.random.choice(len(peakInfo), 2)
        X_orig = peakInfo[sample][:, 0]
        X_orig = X_orig.reshape(-1, 1)
        X = np.append(np.ones_like(X_orig), X_orig, axis=1)
        y = peakInfo[sample][:, 1]
        #   print(len(X), len(y))
        #   print(X,y)
        reg = LinearRegression(fit_intercept=False).fit(X, y)
        reg.score(X, y)
        a = reg.coef_

        count = 0
        inliers = []

        visited = set()

        for point in peakInfo:
            pred = a[0] + a[1] * point[0]

            if np.abs(pred - point[1]) < ransac_threshold and not point[2] in visited:
                visited.add(point[2])
                count += 10 - (pred - point[
                    1]) ** 2  # si el punto esta mas lejos de la recta contribuye un poco menos a la suma

                inliers.append(point)

        count -= np.abs(a[0])  # Intentamos minimizar la fase detectada

        if count > maxCount:
            maxCount = count
            a_opt = a
            inliers_opt = inliers

    inliers_opt_lin = np.array(inliers_opt)

    return inliers_opt_lin[:, 0], inliers_opt_lin[:, 1]

dat = Table.read('../DR3_photogeo_ruwe14_GAL6D.fits', format='fits')
df = dat.to_pandas()

# CROSSING POINTS

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

do_plot = True
# step_z = 2/80
# step_vz = 120/120
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

axisfig = [0.27, 0.2, 0.7, 0.7]
# normalization of the colourscale:
nn = 0.35

# r_range =np.arange(6.5,11, 0.5)
phi_range = np.arange(-4, 4, phi_bin)
r_range = np.arange(10, 10.5, R_bin)
# r_range = 8.28
colors = iter(cm.rainbow(np.linspace(0, 1, len(r_range))))
linear_r2_rec = []
quadratic_r2_rec = []
cubic_r2_rec = []
log_r2_rec = []
radios_recorridos = []

fig, ax = plt.subplots(2, 2,figsize=(12,10))
#gal_radio = 8.1
def calc_histogram_cartesian (df,z_bins,vz_bins ,weights = None):
    if weights == None:
     #   H, xedges, yedges = np.histogram2d(df['Z'].values, df['VZ'].values,bins=[z_bins,vz_bins],range=[rangex,rangey])
        H, xedges, yedges = np.histogram2d(df['Z'].values, df['VZ'].values,bins=[z_bins,vz_bins],range=[rangex,rangey],
                                           normed = mcolors.LogNorm(3))
    elif weights =="Vphi":
        H, xedges, yedges = np.histogram2d(df['Z'].values, df['VZ'].values,
                                   bins=[z_bins,vz_bins], weights = df['Vphi'].values)
     #   H_counts, xedges, yedges = np.histogram2d(df['Z'].values, df['VZ'].values,bins=[z_bins,vz_bins])
        stat=stats.binned_statistic_2d(df['Z'], df['VZ'], df['Vphi'], statistic='mean', bins=[z_bins,vz_bins], range=[rangex,rangey])
        H = stat[0]
     #   H = np.abs(H/H_counts)
    return H, xedges, yedges
def filter_data (df, R_cut, phi_cut, R_bin = 0.8, phi_bin = 3, Vphi_sun = -236):
  #  dfIt = df[(np.abs(df['phi']-phi_cut)<phi_bin) & (df['R']-R_cut<R_bin)]
    dfIt = df[(np.abs(df['phi']-phi_cut)<phi_bin) & (np.abs(df['R']*df['Vphi']-R_cut*Vphi_sun)<np.abs(R_bin*Vphi_sun))]
    return dfIt

gal_radio = 8

c = next(colors)
dfIt = filter_data (df = df,R_cut=gal_radio, phi_cut=0, R_bin =0.5, phi_bin = 10)
print("Number of particles ", len(dfIt))
#fig, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=(18,5), gridspec_kw={'hspace':0,'wspace':0.15})

H,xedges,yedges = calc_histogram_cartesian(dfIt, z_bins,vz_bins, weights = None)
ax[0, 0].cla()
ax[0,0].set_title (f"Rg:{gal_radio:.2f}, phi:{phi_cut}")
sc=ax[0,0].imshow(H.T, origin='lower', cmap='inferno_r',
                          extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                          aspect='auto')

#  sigma = -3.961e-7*len(dfIt) + 2.723
#  sigma = -1e-7*len(dfIt) + 3.3 + 5/(gal_radio)**2
sigma = -1e-7*len(dfIt) + 2.0 + 5/(gal_radio)**2
#   sigma = 2
print(f"para radio {gal_radio}, hay {len(dfIt)} particulas, sigma = {sigma}")

m = feature.canny(np.log10(H),  sigma=sigma)
#  print(m)
z_array = []
vz_array = []
zcentres = 0.5*(xedges[1:]+xedges[:-1])
vzcentres = 0.5*(yedges[1:]+yedges[:-1])
for i, z_i in enumerate(zcentres):
    for j, vz_i in enumerate(vzcentres):
        if m[i,j]== True:
            z_array.append(z_i)
            vz_array.append(vz_i)
im=np.flip(m.T*1.,0)
#    normalization = np.std(dfIt["VZ"]/np.std(dfIt["Z"]))
vz_array = np.array(vz_array)
z_array = np.array(z_array)
z_scale = np.std(dfIt["Z"])
vz_scale = np.std(dfIt["VZ"])
normalization = np.std(dfIt["VZ"])/np.std(dfIt["Z"])
print(normalization)

vz_array_re = np.array(vz_array)/vz_scale
z_array_re = np.array(z_array)/z_scale
r_pre = np.sqrt((dfIt["VZ"]/vz_scale)**2 +(dfIt["Z"]/z_scale)**2)
phi_pre = np.arctan2(dfIt["VZ"]/vz_scale,dfIt["Z"]/z_scale)
p_limit = np.percentile(r_pre, 92)

z_array_filt  = z_array [(np.sqrt(vz_array_re**2+ z_array_re**2) < p_limit)]
vz_array_filt  = vz_array [(np.sqrt(vz_array_re**2+ z_array_re**2) < p_limit)]

z_array_filt_re  = z_array_re [(np.sqrt(vz_array_re**2+ z_array_re**2) <p_limit)]
vz_array_filt_re  = vz_array_re [(np.sqrt(vz_array_re**2+ z_array_re**2) <p_limit)]

rz = np.sqrt(vz_array_filt_re**2 +z_array_filt_re**2)
phiz = np.arctan2(vz_array_filt_re,z_array_filt_re)


phi_re, r_re = unwrap(rz, phiz, wraps = wraps)
X = phi_re
y = r_re

X = X.reshape(-1, 1)
y = y.reshape(-1, 1)



regr = LinearRegression()
quadratic = PolynomialFeatures(degree=2)
cubic = PolynomialFeatures(degree=3)
X_quad = quadratic.fit_transform(X)
X_cubic = cubic.fit_transform(X)

X_fit = np.arange(X.min(), X.max()+1, 0.1)[:, np.newaxis]

regr = regr.fit(X, y)
y_lin_fit = regr.predict(X_fit)
linear_r2 = r2_score(y, regr.predict(X))
print(f"reg coef is {regr.coef_}")
if regr.coef_ < 0:
    phi_re, r_re = unwrap(rz, phiz, wraps = 2)
    X = phi_re
    y = r_re

    X = X.reshape(-1, 1)
    y = y.reshape(-1, 1)


    regr = LinearRegression()
    quadratic = PolynomialFeatures(degree=2)
    cubic = PolynomialFeatures(degree=3)
    X_quad = quadratic.fit_transform(X)
    X_cubic = cubic.fit_transform(X)

    X_fit = np.arange(X.min(), X.max()+1, 0.1)[:, np.newaxis]

    regr = regr.fit(X, y)
    y_lin_fit = regr.predict(X_fit)
    linear_r2 = r2_score(y, regr.predict(X))


f = interp1d(phi_re, r_re,kind='nearest')
# order = 3
# do inter/extrapolation
#f = InterpolatedUnivariateSpline(phi_re, r_re, k=order)
#y = s(x)

regr = regr.fit(X, np.log(y))
y_log_fit = regr.predict(X_fit)
log_r2 = r2_score(np.log(y), regr.predict(X))

regr = regr.fit(X_quad, y)
y_quad_fit = regr.predict(quadratic.fit_transform(X_fit))
quadratic_r2 = r2_score(y, regr.predict(X_quad))

regr = regr.fit(X_cubic, y)
y_cubic_fit = regr.predict(cubic.fit_transform(X_fit))
cubic_r2 = r2_score(y, regr.predict(X_cubic))
ax[0, 1].cla()


ax[0,1].scatter(z_array_filt, vz_array_filt,color = c,marker = "o",s = 5, alpha = 0.3, label = f"Rg ={gal_radio:.2f} ")
ax[0,1].set_xlim(-0.9,0.9)
ax[0,1].set_ylim(-50,50)
ax[0,1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
#ax[0,2].scatter(phiz, rz, color = c,marker = "o",s = 5, alpha = 1)
ax[1,1].scatter(phiz, rz, color = c,marker = "o",s = 5, alpha = 0.3)
ax[1,1].scatter(phiz +2*np.pi, rz, color = c,marker = "o",s = 5, alpha = 0.3)
ax[1,1].scatter(phiz + 4*np.pi, rz, color = c,marker = "o",s = 5, alpha = 0.3)
ax[1,1].set_xlim(-np.pi,5*np.pi)
ax[1,1].set_ylim(0,2.5)
###We compare nonlinear regression here with different power
#phi_re, r_re

ax[1, 0].cla()
ax[1,0].scatter(X, y, label='training points', color='blue')

ax[1,0].plot(X_fit, y_lin_fit,
         label='linear (d=1), $R^2=%.2f$' % linear_r2,
         color='blue',
         lw=2,
         linestyle=':', alpha = 0.2)

ax[1,1].plot(X_fit, y_lin_fit,
         label='linear (d=1), $R^2=%.2f$' % linear_r2,
         color='blue',
         lw=2,
         linestyle=':', alpha = 1)


ax[1,0].plot(X_fit, y_quad_fit,
         label='quadratic (d=2), $R^2=%.2f$' % quadratic_r2,
         color='red',  lw=2, linestyle='-', alpha = 0.2)

ax[1,0].plot(X_fit, y_cubic_fit,
         label='cubic (d=3), $R^2=%.2f$' % cubic_r2,
         color='green', lw=2, linestyle='--')

ax[1,0].plot(X_fit, np.exp(y_log_fit),
         label='Logarithmic, $R^2=%.2f$' % log_r2,
         color='purple',
         lw=2,
         linestyle='--')
ax[1,0].legend(loc='upper left')

linear_r2_rec.append(linear_r2)
quadratic_r2_rec.append(quadratic_r2)
cubic_r2_rec.append(cubic_r2)
log_r2_rec.append(log_r2)


#   ax[1,1].legend()
x_predict = np.arange(phi_re.min(), phi_re.max(), 0.1)
y_predict = f(x_predict)

plt.show()