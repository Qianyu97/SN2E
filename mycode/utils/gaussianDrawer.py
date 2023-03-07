import numpy as np
import torch
from matplotlib.patches import Ellipse, Circle
import matplotlib.pyplot as plt
from matplotlib import transforms
from random import sample

from mycode.models.base import Module 
from mycode.utils import prepare
from finaldata import FinalData
from config import DatapathArg, TestArg
from mycode.utils.treeunit import NodeUnit

class GaussianDrawer():
    def __init__(self, dataset:FinalData, model:Module):
        self.dataset = dataset
        self.model = model
        self.showSamples = self.geneShowSamples(
            dataset.finedata.rawdata.basedict,
            TestArg.num_showpicture)
        self.reduceDim = 2
    
    def geneShowSamples(self, basedict:dict, shownum = 10):
        showsamples = list()
        for conceptname in sample(basedict.keys(), shownum):
            concept:NodeUnit = basedict[conceptname]
            showson     = [concept.name] + list(concept.sons)
            showattr    = [concept.name] + list(concept.attributes)
            showsamples.append(showson)
            showsamples.append(showattr)
        return showsamples

    def drawSamples(self):
        for i, sample in enumerate(self.showSamples):
            self.drawOneSample(sample, sample[0] + str(i+1))

    def drawOneSample(self, conceptlist, label = 'default'):
        plt.switch_backend('agg')
        indexConcepts = self.dataset.indexconvert(conceptlist)
        embedding = self.model.lookupEmbedding(torch.tensor(indexConcepts))
        reducedMeans, reducedVariances = self.PCA_Gaussians(embedding)
        figure , ax = plt.subplots()
        self.gaussians_ellipse(indexConcepts, reducedMeans, reducedVariances, ax, facecolor = 'blue')
        plt.axis('scaled')
        plt.axis('equal')   #changes limits of x or y axis so that equal increments of x and y have the same length
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig(DatapathArg.path_picture + label + ".jpg")
    
    def PCA_Gaussians(self, embedding):
        [means, inVariances] = embedding
        means, variances = means.unsqueeze(-1), torch.diag_embed(1/inVariances)

        MeanAve              = means.mean()
        variancePCA         = ((means - MeanAve) @ (means - MeanAve).transpose(1, 2)).mean(0)

        (evals,evecs) = torch.eig(variancePCA, eigenvectors=True)
        eTopkIndex         = evals[:,0].argsort(descending=True)[ : self.reduceDim]
        reducedEvecs       = evecs[: , eTopkIndex]
        
        reducedMeans        = reducedEvecs.T @ means
        reducedVariances    = reducedEvecs.T @ variances @ reducedEvecs
        return reducedMeans, reducedVariances


    def gaussians_ellipse(self, conceptlist, mean, variance, ax, n_std=1, facecolor='yellow',**kwargs):
        """
        Create a plot of the covariance confidence ellipse of *x* and *y*.

        Parameters
        ----------
        x, y : array-like, shape (n, )
            input data.

        ax : matplotlib.axes.Axes
            The axes object to draw the ellipse into.

        n_std : float
            The number of standard deviations to determine the ellipse's radiuses.

        **kwargs
            Forwarded to `~matplotlib.patches.Ellipse`

        Returns
        -------
        matplotlib.patches.Ellipse
        """

        for i in range(mean.shape[0]):
            cov = variance[i]
            pearson = cov[0, 1]/torch.sqrt(cov[0, 0] * cov[1, 1])
            pearson = pearson if pearson < 1 else 1
            # Using a special case to obtain the eigenvalues of this
            # two-dimensionl dataset.
            ell_radius_x = torch.sqrt(1 + pearson).item()
            ell_radius_y = torch.sqrt(1 - pearson).item()
            ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                                facecolor = facecolor, alpha=0.3)

            # Calculating the stdandard deviation of x from
            # the squareroot of the variance and multiplying
            # with the given number of standard deviations.
            scale_x = torch.sqrt(cov[0, 0]).item() * n_std
            mean_x = mean[i, 0].item()

            # calculating the stdandard deviation of y ...
            scale_y = torch.sqrt(cov[1, 1]).item() * n_std
            mean_y = mean[i, 1].item()

            transf = transforms.Affine2D() \
                .rotate_deg(45) \
                .scale(scale_x, scale_y) \
                .translate(mean_x, mean_y)
            ellipse.set_transform(transf + ax.transData)
            ax.add_patch(ellipse)
            showinfo = self.dataset.indexconvert(conceptlist[i], 'num2str')
            ax.annotate(showinfo, xytext = (mean_x, mean_y), xy = (mean_x, mean_y ) )
        return ax



if __name__ == "__main__":
    dataset    = FinalData(DatapathArg.path_rawdata, DatapathArg.path_indexdict)
    model      = prepare.prepareModel(ifLoadModel=True)
    drawer     = GaussianDrawer(dataset, model)
    drawer.drawSamples()
    
