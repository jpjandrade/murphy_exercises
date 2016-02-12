# coding: utf-8

from matplotlib.patches import Ellipse

data = loadtxt('heightWeightData.txt', delimiter=',', dtype = int)
data_male = data[data[:,0] == 1]
data_male = data_male[:,1:]

mu_male = average(data_male, axis = 0)
sigma_male = np.cov(data_male.T, bias = 1)

l, v = np.linalg.eig(sigma_male)

fig = figure()
ax = fig.add_subplot(111)
ax.scatter(data_male[:,0], data_male[:,1])
ell = Ellipse(xy = mu_male, width = 4*sqrt(l[0]), height = 4*sqrt(l[1]), angle = np.rad2deg(arccos(v[0,0])))
ell.set_facecolor('none')
ax.add_artist(ell)
fig.savefig('test.png')
