import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
from matplotlib import transforms
from sklearn.metrics import roc_curve, auc

from utils.unit import Embedding

def plot_roc(y_true:torch.Tensor, y_score:torch.Tensor, title="ROC Curve", savepath:str=None):
    figure , ax = plt.subplots()
    plt.switch_backend('agg')
    y_true = y_true.detach().cpu().numpy()
    y_score = y_score.detach().cpu().numpy()
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.5)
    if savepath is not None:
        plt.savefig(savepath)
    else:
        plt.show()
    return roc_auc

def gaussians_ellipse(
        conceptlist:list[str], 
        embedding:Embedding,  
        n_std=1, facecolor = 'yellow',
        saveDirectory = None 
        ):
    '''
    embedding       : [num0, dim]
    mean, variance  : [num0,   2]
    '''
    mean, variance = PCA_Gaussians(embedding, target_dim=2)
    assert mean.shape[0] == variance.shape[0] == len(conceptlist)
    figure , ax = plt.subplots()
    plt.switch_backend('agg')
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
        showinfo = conceptlist[i]
        ax.annotate(showinfo, xytext = (mean_x, mean_y), xy = (mean_x, mean_y ) )
    plt.axis('scaled')
    plt.axis('equal')   #changes limits of x or y axis so that equal increments of x and y have the same length
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(saveDirectory)
    return ax

def PCA_Gaussians(embedding:Embedding, target_dim=2)->Embedding:
    'embedding: [num0, dim]'
    means, invariances = embedding
    means = means.unsqueeze(-2)
    variances = torch.diag_embed(invariances.reciprocal())
    MeanAve     = means.mean(dim=0)
    a           = (means - MeanAve)#*(embedding.v.sum(-2).sqrt().unsqueeze(-1))
    variancePCA = ((a.transpose(1, 2) @ a)).mean(0)
    evals, evecs = torch.linalg.eigh(variancePCA)
    eTopkIndex  = evals.real.argsort(descending=True)[ : target_dim]
    reducedEvecs    = evecs[: , eTopkIndex].real
    reducedMeans    =means @ reducedEvecs
    reducedVariances= reducedEvecs.T @ variances @ reducedEvecs
    return reducedMeans, reducedVariances

