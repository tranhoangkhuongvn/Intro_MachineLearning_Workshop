import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap


from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):
    # Adapted from Hands-on Machine Learning with scikit-learn 2ed
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

def plot_decision_boundary(clf, X, y, axes=[0, 7.5, 0, 3], iris=True, legend=False, plot_training=True):
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)
    if not iris:
        custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
        plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
    if plot_training:
        plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", label="Iris setosa")
        plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", label="Iris versicolor")
        plt.plot(X[:, 0][y==2], X[:, 1][y==2], "g^", label="Iris virginica")
        plt.axis(axes)
    if iris:
        plt.xlabel("Petal length", fontsize=14)
        plt.ylabel("Petal width", fontsize=14)
    else:
        plt.xlabel(r"$x_1$", fontsize=18)
        plt.ylabel(r"$x_2$", fontsize=18, rotation=0)
    if legend:
        plt.legend(loc="lower right", fontsize=14)
        

## Visualise the decision boundary
def plot_2d_boundary(features, targets, intercept, coef):
    # Identify class
    pos = np.where(targets==1)[0]
    neg = np.where(targets==0)[0]
    # Retrieve the model parameters.
    b = intercept[0]
    w1, w2 = coef.T
    # Calculate the intercept and gradient of the decision boundary.
    c = -b/w2
    m = -w1/w2

    # Plot the data and the classification with the decision boundary.
    xmax, ymax = features.max(axis=0)
    xmin, ymin = features.min(axis=0)
    xd = np.array([xmin, xmax])
    yd = m*xd + c
    plt.figure(figsize=(10,5))
    plt.plot(xd, yd, 'k', lw=1, ls='--')



    #plt.plot(points_x, points_y)
    # Plot examples
    plt.plot(features[pos, 0], features[pos, 1], 'b+', label='Admitted')
    plt.plot(features[neg, 0], features[neg, 1], 'yo', label='Not admitted')
    plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
    plt.show()
    

def plot_dataset(X, y, axes):
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "bs")
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "g^")
    plt.axis(axes)
    plt.grid(True, which='both')
    plt.xlabel(r"$x_1$", fontsize=20)
    plt.ylabel(r"$x_2$", fontsize=20, rotation=0)
    


def plot_3D(X, figsize=(10,5)):
    # Adapted from Hands-on Machine Learning with scikit-learn 2ed
    fig = plt.figure(figsize=figsize)
    m = X.shape[0]
    ax = fig.add_subplot(111, projection='3d')

    axes = [-1.8, 1.8, -1.3, 1.3, -1.0, 1.0]

    # Hard code the component to make the visualisation of the plane
    x1s = np.linspace(axes[0], axes[1], 10)
    x2s = np.linspace(axes[2], axes[3], 10)
    x1, x2 = np.meshgrid(x1s, x2s)
    C = np.array([[-0.93636116, -0.29854881, -0.18465208],
       [ 0.34027485, -0.90119108, -0.2684542 ]])
    
    z = np.array([[-0.59627063, -0.55977498, -0.52327932, -0.48678367, -0.45028801,
        -0.41379236, -0.37729671, -0.34080105, -0.3043054 , -0.26780974],
       [-0.5002617 , -0.46376605, -0.42727039, -0.39077474, -0.35427908,
        -0.31778343, -0.28128778, -0.24479212, -0.20829647, -0.17180081],
       [-0.40425277, -0.36775712, -0.33126146, -0.29476581, -0.25827015,
        -0.2217745 , -0.18527884, -0.14878319, -0.11228754, -0.07579188],
       [-0.30824384, -0.27174819, -0.23525253, -0.19875688, -0.16226122,
        -0.12576557, -0.08926991, -0.05277426, -0.01627861,  0.02021705],
       [-0.21223491, -0.17573926, -0.1392436 , -0.10274795, -0.06625229,
        -0.02975664,  0.00673902,  0.04323467,  0.07973032,  0.11622598],
       [-0.11622598, -0.07973032, -0.04323467, -0.00673902,  0.02975664,
         0.06625229,  0.10274795,  0.1392436 ,  0.17573926,  0.21223491],
       [-0.02021705,  0.01627861,  0.05277426,  0.08926991,  0.12576557,
         0.16226122,  0.19875688,  0.23525253,  0.27174819,  0.30824384],
       [ 0.07579188,  0.11228754,  0.14878319,  0.18527884,  0.2217745 ,
         0.25827015,  0.29476581,  0.33126146,  0.36775712,  0.40425277],
       [ 0.17180081,  0.20829647,  0.24479212,  0.28128778,  0.31778343,
         0.35427908,  0.39077474,  0.42727039,  0.46376605,  0.5002617 ],
       [ 0.26780974,  0.3043054 ,  0.34080105,  0.37729671,  0.41379236,
         0.45028801,  0.48678367,  0.52327932,  0.55977498,  0.59627063]])
    

    ax.plot(X[:, 0], X[:, 1], X[:, 2], "bo", alpha=0.5)

    ax.plot_surface(x1, x2, z, alpha=0.2, color="k")
    np.linalg.norm(C, axis=0)
    ax.add_artist(Arrow3D([0, C[0, 0]],[0, C[0, 1]],[0, C[0, 2]], mutation_scale=15, lw=1, arrowstyle="-|>", color="r"))
    ax.add_artist(Arrow3D([0, C[1, 0]],[0, C[1, 1]],[0, C[1, 2]], mutation_scale=15, lw=1, arrowstyle="-|>", color="g"))
    ax.plot([0], [0], [0], "k.")


    ax.set_xlabel("$x_1$", fontsize=18, labelpad=10)
    ax.set_ylabel("$x_2$", fontsize=18, labelpad=10)
    ax.set_zlabel("$x_3$", fontsize=18, labelpad=10)
    ax.set_xlim(axes[0:2])
    ax.set_ylim(axes[2:4])
    ax.set_zlim(axes[4:6])


    plt.show()
    

def plot_cumsum(cumsum, d, figsize=(10, 6)):
    plt.figure(figsize=figsize)
    plt.plot(cumsum, linewidth=3)
    plt.axis([0, 400, 0, 1])
    plt.xlabel("Dimensions")
    plt.ylabel("Explained Variance")
    plt.plot([d, d], [0, 0.95], "k:")
    plt.plot([0, d], [0.95, 0.95], "k:")
    plt.plot(d, 0.95, "ko")
    plt.annotate("Elbow", xy=(65, 0.85), xytext=(70, 0.7),
                 arrowprops=dict(arrowstyle="->"), fontsize=16)
    plt.grid(True)
    plt.show()
    

def plot_k_clusters(X, estimator, n_clusters=3):

    
    titles = str(n_clusters) + " clusters"

    fig = plt.figure(figsize=(15, 10))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

    labels = estimator.labels_

    ax.scatter(X[:, 3], X[:, 0], X[:, 2],
               c=labels.astype(np.float), edgecolor='k', s=60)

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('Petal width')
    ax.set_ylabel('Sepal length')
    ax.set_zlabel('Petal length')
    ax.set_title(titles)
    ax.dist = 12
    plt.show()