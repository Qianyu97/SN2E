import numpy as np
import torch
from matplotlib.patches import Ellipse, Circle
import matplotlib.pyplot as plt
from matplotlib import transforms
import random

from config import PathArg, TrainArg, TestArg
from config_model import SN2E_Arg
from models.SN2E import SN2E
from utils import initModule
from finaldata import FinalData

from utils.treeunit import NodeUnit, AttributeUnit, Unit
from utils.embedding import Embedding, EmbeddingOperator
from utils import initModule

class GaussianDrawer():
    def __init__(self, dataset:FinalData, model:SN2E, pictureDirectory:str=''):
        self.dataset = dataset
        self.baselist = dataset.defiList[1:]
        self.model = model
        self.pictureDirectory = pictureDirectory
    
    def drawSamples(self, shownum = 10)->list[NodeUnit]:
        showTargets:list[NodeUnit] = list()
        showTargets = random.sample(self.baselist, shownum)
        for i, sample in enumerate(showTargets):
            self.drawOneConcept(sample.name)
    
    def drawOneConcept(self, name:str):
        concept = self.dataset.defiUnitDict[name]
        attrlist = list(concept.attributes)
        children = list(concept.children)
        father = concept.father

        index0 = concept.index
        attrlist_idx = [attr.index for attr in attrlist]
        father_idx = father.index
        children_idx = [child.index for child in children]
        
        conceptEmbedding = self.model.lookupDefiEmbedding(index0, ifdetach=True)
        attrEmbedding = self.model.lookupAttrEmbedding(attrlist_idx, ifdetach=True)
        fatherEmbedding = self.model.lookupDefiEmbedding(father_idx, ifdetach=True)
        childrenEmbedding = self.model.lookupDefiEmbedding(children_idx, ifdetach=True)
        self.drawAttributeSample(concept, attrlist, father,  label = f"{concept.name} attributes")
        self.drawChildrenSample(concept, children, label = f"{concept.name} children")

    def drawAttributeSample(self, concept:NodeUnit, attributes:list[Unit], father:NodeUnit, label = 'default'):
        index0 = concept.index
        indexA = [item.index for item in attributes]
        indexF = father.index
        e_0 = self.model.lookupDefiEmbedding(index0, ifdetach=True)
        e_a = self.model.lookupAttrEmbedding(indexA, ifdetach=True)
        e_f = self.model.lookupDefiEmbedding(indexF, ifdetach=True)
        e_u:Embedding = EmbeddingOperator.cat([e_0, e_a, e_f])
        reducedEmbedding = self.PCA_Gaussians(e_u)
        figure , ax = plt.subplots()
        plt.switch_backend('agg')
        self.gaussians_ellipse([concept] + attributes + [father], reducedEmbedding, ax, facecolor = 'blue')
        plt.axis('scaled')
        plt.axis('equal')   #changes limits of x or y axis so that equal increments of x and y have the same length
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig(self.pictureDirectory + label + ".jpg")

    def drawChildrenSample(self, concept:NodeUnit, children:list[Unit], label = 'default'):
        if len(children) == 0:
            return
        index0 = concept.index
        children_idx = [item.index for item in children]
        e_0 = self.model.lookupDefiEmbedding(index0, ifdetach=True)
        e_a = self.model.lookupDefiEmbedding(children_idx, ifdetach=True)
        e_u:Embedding = EmbeddingOperator.cat([e_0, e_a])
        reducedEmbedding = self.PCA_Gaussians(e_u)
        figure , ax = plt.subplots()
        plt.switch_backend('agg')
        self.gaussians_ellipse([concept] + children, reducedEmbedding, ax, facecolor = 'blue')
        plt.axis('scaled')
        plt.axis('equal')   #changes limits of x or y axis so that equal increments of x and y have the same length
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig(self.pictureDirectory + label + ".jpg")
    
    def PCA_Gaussians(self, embedding:Embedding)->Embedding:
        means, invariances = embedding
        means = means.unsqueeze(-2)
        variances = torch.diag_embed(invariances.reciprocal())
        MeanAve     = means.mean(dim=0)
        a           = (means - MeanAve)#*(embedding.v.sum(-2).sqrt().unsqueeze(-1))
        variancePCA = ((a @ a.transpose(1, 2)) + variances).mean(0)
        evals, evecs = torch.linalg.eigh(variancePCA)
        eTopkIndex  = evals.real.argsort(descending=True)[ : 2]
        reducedEvecs    = evecs[: , eTopkIndex].real
        reducedMeans    =means @ reducedEvecs
        reducedVariances= reducedEvecs.T @ variances @ reducedEvecs
        return reducedMeans, reducedVariances

    def gaussians_ellipse(self, conceptlist:list[Unit], embedding, ax, n_std=1, facecolor='yellow',**kwargs):
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
        mean, variance = embedding
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
            mean_x = mean[i, 0, 0].item()

            # calculating the stdandard deviation of y ...
            scale_y = torch.sqrt(cov[1, 1]).item() * n_std
            mean_y = mean[i, 0, 1].item()

            transf = transforms.Affine2D() \
                .rotate_deg(45) \
                .scale(scale_x, scale_y) \
                .translate(mean_x, mean_y)
            ellipse.set_transform(transf + ax.transData)
            ax.add_patch(ellipse)
            showinfo = conceptlist[i].name
            ax.annotate(showinfo, xytext = (mean_x, mean_y), xy = (mean_x, mean_y ) )
        return ax


if __name__ == "__main__":
    datapathArg = PathArg()
    modelArg = SN2E_Arg()
    testArg = TestArg()
    dataset = FinalData(datapathArg.dataDirectory, modelArg)
    model = initModule.initModel(modelArg, ifloadmodel=True, modelDir=datapathArg.modelDirectory, usegpu=testArg.usegpu, gpunum=testArg.gpunum)
    drawer = GaussianDrawer(dataset, model, datapathArg.pictureDirectory)
    drawer.drawSamples(3)
    a=0