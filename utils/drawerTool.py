import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
from matplotlib import transforms
from sklearn.metrics import roc_curve, precision_recall_curve, auc

from utils.unit import Embedding

plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
})

def plot_roc(y_true:torch.Tensor, y_score:torch.Tensor, title="ROC Curve", savepath:str=None, draw=False):
    figure , ax = plt.subplots()
    plt.switch_backend('agg')
    y_true = y_true.detach().cpu().numpy()
    y_score = y_score.detach().cpu().numpy()
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    if draw:
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

def plot_pr_auc(
    y_true: torch.Tensor,
    y_score: torch.Tensor,
    title="Precision-Recall Curve",
    savepath: str = None,
    draw=False
    ):
    y_true = y_true.detach().cpu().numpy()
    y_score = y_score.detach().cpu().numpy()
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    if draw ==True:
        plt.switch_backend('agg')
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(recall, precision, label=f"PR-AUC = {pr_auc:.4f}")

        # 随机基线（正类比例）
        positive_rate = y_true.mean()
        ax.hlines(
            y=positive_rate,
            xmin=0,
            xmax=1,
            linestyles="--",
            linewidth=1,
            label=f"Random = {positive_rate:.4f}"
        )

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(title)
        ax.legend(loc="lower left")
        ax.grid(True, linestyle="--", alpha=0.5)

        if savepath is not None:
            plt.savefig(savepath)
        else:
            plt.show()
    return pr_auc

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
    def PCA_Gaussians(embedding:Embedding, target_dim=2)->Embedding:
        'embedding: [num0, dim]'
        means, invrriances = embedding
        means = means.unsqueeze(-2)
        variances = torch.diag_embed(invrriances.reciprocal())
        MeanAve     = means.mean(dim=0)
        a           = (means - MeanAve)#/invrriances.reciprocal().sum(-2)#*(embedding.v.sum(-2).sqrt().unsqueeze(-1))
        variancePCA = ((a.transpose(1, 2) @ a)).mean(0)
        evals, evecs = torch.linalg.eigh(variancePCA)
        eTopkIndex  = evals.real.argsort(descending=True)[ : target_dim]
        reducedEvecs    = evecs[: , eTopkIndex].real
        reducedMeans    =means @ reducedEvecs
        reducedVariances= reducedEvecs.T @ variances @ reducedEvecs
        return reducedMeans, reducedVariances
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
    plt.axis('off')
    #plt.xlabel('x')
    #plt.ylabel('y')
    plt.savefig(saveDirectory)
    return ax

def plot_grouped_bars(
    scores: dict,
    methods_order=None,
    metrics_order=None,
    title=None,
    max_bar_height: float = 1.0,
    group_width: float = 0.85,
    show_values: bool = False,
    value_fmt: str = "{:.3f}",
    ax=None,
):
    """
    竖向并排紧贴柱状图：
    - x 轴: 指标(H@1 / F1 / AUC …）
    - 颜色: 方法(Ours / KL / Wasserstein …）
    - 每个指标内部：最大值柱高 = max_bar_height,其余按比例缩放
    使用示例：
        scores = {
            "H@1": {"Ours": 0.459, "KL": 0.411, "Wasserstein": 0.393},
            "F1":  {"Ours": 0.891, "KL": 0.859, "Wasserstein": 0.784},
            "AUC": {"Ours": 0.920, "KL": 0.900, "Wasserstein": 0.880},
        }
        fig, ax = plot_grouped_bars(
            scores,
            methods_order=["Ours", "KL", "Wasserstein"],
            max_bar_height=1.0
        )
        plt.show()
        fig.savefig("grouped_bars.png", dpi=300)
    """
    # -------- 指标 & 方法顺序 --------
    if metrics_order is None:
        metrics = list(scores.keys())
    else:
        metrics = list(metrics_order)

    all_methods = []
    for m in metrics:
        all_methods.extend(scores.get(m, {}).keys())
    unique_methods = list(dict.fromkeys(all_methods))  # 保序去重

    if methods_order is None:
        methods = unique_methods
    else:
        methods = list(methods_order) + [
            m for m in unique_methods if m not in methods_order
        ]

    M = len(metrics)
    K = len(methods)

    # -------- 每个指标内部归一化 --------
    norm = {metric: {} for metric in metrics}
    for metric in metrics:
        vals = [scores[metric][m] for m in methods if m in scores[metric]]
        vmax = max(vals)
        for m in methods:
            if m in scores[metric]:
                norm[metric][m] = (
                    scores[metric][m] / vmax * max_bar_height
                    if vmax != 0 else 0.0
                )

    # -------- 画图 --------
    if ax is None:
        fig, ax = plt.subplots(figsize=(1.8 * M, 4))
    else:
        fig = ax.figure

    x = np.arange(M)
    bar_width = group_width / K
    offsets = (np.arange(K) - (K - 1) / 2) * bar_width

    for i, method in enumerate(methods):
        xs = x + offsets[i]
        heights = [
            norm[metric].get(method, 0.0) for metric in metrics
        ]

        bars = ax.bar(xs, heights, width=bar_width, label=method)

        if show_values:
            for j, b in enumerate(bars):
                if method not in scores[metrics[j]]:
                    continue
                v = scores[metrics[j]][method]
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    b.get_height() + max_bar_height * 0.02,
                    value_fmt.format(v),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
    # -------- 轴 & 美化 --------
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, max_bar_height * 1.05)
    ax.set_yticks(np.linspace(0, 1.0, 3))
    ax.set_ylabel(f"Relative score")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_title(title)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    return fig, ax

def plot_f1_curves(
    x,
    y_dict,
    xlim=(-5, 5),
    ylim=(0, 1),
    xlabel=None,
    ylabel="F1 score",
    title=None,
    ax=None,
):
    """
    绘制多条 F1 score 折线图

    使用示例:
    x = np.linspace(-5, 5, 11)
    y_dict = {
        r"$\delta_{\mathrm{obj}}$": np.array([0.60, 0.63, 0.71, 0.75, 0.78, 0.76, 0.73, 0.69,  0.67, 0.65, 0.62]),
        r"$h_{\mathrm{obj}}$":   np.array([0.55, 0.58, 0.61, 0.63, 0.69, 0.75, 0.79, 0.76, 0.68, 0.60, 0.57]),
    }
    fig, ax = plot_f1_curves(
        x,
        y_dict,
        xlabel="The value of margin",
        title=None
    )
    plt.show()
    fig.savefig("f1_curves.png", dpi=300)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.figure
    for name, y in y_dict.items():
        ax.plot(
            x,
            y,
            marker=None,
            linewidth=2,
            label=name,
        )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks(np.linspace(*xlim, 5))
    ax.set_yticks(np.linspace(*ylim, 6))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="lower right",frameon=True)
    fig.tight_layout()
    return fig, ax

if __name__ == "__main__":
    x = np.linspace(-5, 5, 11)
    y_dict = {
        r"$\delta_{\mathrm{obj}}$": np.array([0.60, 0.63, 0.71, 0.75, 0.78, 0.76, 0.73, 0.69,  0.67, 0.65, 0.62]),
        r"$h_{\mathrm{obj}}$":   np.array([0.55, 0.58, 0.61, 0.63, 0.69, 0.75, 0.79, 0.76, 0.68, 0.60, 0.57]),
    }
    fig, ax = plot_f1_curves(
        x,
        y_dict,
        title=None
    )
    plt.show()
    fig.savefig("f1_curves.png", dpi=300)



