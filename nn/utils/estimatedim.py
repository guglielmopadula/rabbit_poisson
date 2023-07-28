from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import skdim
pca=PCA()
data=np.load("data/data.npy")
pca.fit(data)
y=np.cumsum(pca.explained_variance_ratio_)
print(np.argmin(np.abs(y-0.999)))
print(np.linalg.norm(data-pca.inverse_transform(pca.transform(data)))/np.linalg.norm(data))
print("twonn dim is", skdim.id.TwoNN().fit(data).dimension_)
print("mom dim is", skdim.id.MOM().fit(data).dimension_)
print("tle dim is", skdim.id.TLE().fit(data).dimension_)

