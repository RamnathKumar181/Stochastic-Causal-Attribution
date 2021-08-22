import numpy as np
import matplotlib.pyplot as plt
import copy
import torch.autograd as autograd
import torch
from utils import mean_l1_norm
from matplotlib.colors import LogNorm


def compute_ace_quantile(X, model_common, fc, num_c=5, num_alpha=1000,
                         title="test"):
    final = []
    cov = np.cov(X, rowvar=False)
    means = np.mean(X, axis=0)
    cov = np.array(cov)
    mean_vector = np.array(means)
    for t in range(0, num_c):
        expectation_do_x = []
        inp = copy.deepcopy(mean_vector)
        for x in np.linspace(0, 1, num_alpha):
            inp[t] = x
            input_torchvar = autograd.Variable(torch.FloatTensor(inp),
                                               requires_grad=True)
            output = fc(model_common(input_torchvar))
            o1 = output.data.cpu()
            val = o1.numpy()[0]
            grad_mask_gradient = torch.zeros(1)
            grad_mask_gradient[0] = 1.0
            first_grads = torch.autograd.grad(output.cpu(), input_torchvar.cpu(
            ), grad_outputs=grad_mask_gradient, retain_graph=True,
                create_graph=True)
            for dimension in range(0, num_c):  # Tr(Hessian*Covariance)
                if dimension == t:
                    continue
                temp_cov = copy.deepcopy(cov)
                temp_cov[dimension][t] = 0.0
                grad_mask_hessian = torch.zeros(num_c)
                grad_mask_hessian[dimension] = 1.0
                # calculating the hessian
                hessian = torch.autograd.grad(
                    first_grads, input_torchvar,
                    grad_outputs=grad_mask_hessian,
                    retain_graph=True, create_graph=False)
                # adding second term in interventional expectation
                val += np.sum(0.5*hessian[0].data.numpy()*temp_cov[dimension])
            # append interventional expectation for given interventional value
            expectation_do_x.append(val)
        final.append(np.array(expectation_do_x) -
                     np.mean(np.array(expectation_do_x)))
    return final


def plot_sace_quantile(x,
                       net_common,
                       net_upper,
                       net_middle,
                       net_lower,
                       title,
                       filename,
                       num_c=5,
                       extend_plot=False,
                       feature_names=None):
    plt.figure(figsize=(10, 10))
    num_alpha = 1000
    causal_effect_middle = compute_ace_quantile(X=x.numpy(),
                                                model_common=net_common,
                                                num_c=num_c,
                                                fc=net_middle)
    causal_effect_lower = compute_ace_quantile(X=x.numpy(),
                                               model_common=net_common,
                                               num_c=num_c,
                                               fc=net_lower)
    causal_effect_upper = compute_ace_quantile(X=x.numpy(),
                                               model_common=net_common,
                                               num_c=num_c,
                                               fc=net_upper)
    # plt.title(title, fontsize=20)
    plt.xlabel('Intervention Value (alpha)', fontsize=26)
    plt.ylabel('Causal Attributions (ACE)', fontsize=26)
    col = {0: "b", 1: "g", 2: "r", 3: "c", 4: "y", 5: "m", 6: '#A52A2A',
           7: "#DAA520", 8: "#6B8E23", 9: "#4B0082", 10: "#BA55D3",
           11: "#DDA0DD", 12: "#8B4513"}
    for t in range(0, num_c):
        if feature_names is None:
            plt.plot(np.linspace(0, 1, num_alpha),
                     causal_effect_middle[t], label=r'$x_{}$'.format(t+1),
                     color=col[t])
        else:
            plt.plot(np.linspace(0, 1, num_alpha),
                     causal_effect_middle[t], label=feature_names[t],
                     color=col[t])
        plt.fill_between(np.linspace(0, 1, num_alpha), causal_effect_middle[t],
                         causal_effect_upper[t], color=col[t], alpha=0.3)
        plt.fill_between(np.linspace(0, 1, num_alpha), causal_effect_middle[t],
                         causal_effect_lower[t], color=col[t], alpha=0.3)
    if extend_plot:
        lg = plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5),
                        fontsize=22)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.tight_layout()
        plt.savefig(f'{filename}.pdf', bbox_extra_artists=(lg,),
                    bbox_inches='tight', dpi=300)
        plt.savefig(f'{filename}.pdf', dpi=300)
    else:
        plt.legend(fontsize=22)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.savefig(f'{filename}.pdf')
    plt.show()


def plot_sace_quantile_saliency_map(x,
                                    net_common,
                                    net_upper,
                                    net_middle,
                                    net_lower,
                                    title,
                                    filename,
                                    num_c=5,
                                    feature_names=None):
    causal_effect_middle = compute_ace_quantile(X=x.numpy(),
                                                model_common=net_common,
                                                num_c=num_c,
                                                fc=net_middle)
    causal_effect_lower = compute_ace_quantile(X=x.numpy(),
                                               model_common=net_common,
                                               num_c=num_c,
                                               fc=net_lower)
    causal_effect_upper = compute_ace_quantile(X=x.numpy(),
                                               model_common=net_common,
                                               num_c=num_c,
                                               fc=net_upper)
    bin = 50
    subarr_middle = [[causal_effect_middle[j][i*bin:i*bin+20]
                      for i in range(20)] for j in range(num_c)]
    subarr_upper = [[causal_effect_upper[j][i*bin:i*bin+20]
                     for i in range(20)] for j in range(num_c)]
    subarr_lower = [[causal_effect_lower[j][i*bin:i*bin+20]
                     for i in range(20)] for j in range(num_c)]
    final = []
    std = []
    for i in range(num_c):
        mean = mean_l1_norm(subarr_middle[i])
        lower = mean_l1_norm(subarr_lower[i])
        upper = mean_l1_norm(subarr_upper[i])
        mean_std = [(abs(lower[i]-mean[i])+abs(upper[i]-mean[i]))/2
                    for i in range(len(mean))]
        final.append(mean)
        std.append(mean_std)

    mean_imp = [np.mean(arr) for arr in final]
    print(f"Most important feature: {(-np.array(mean_imp)).argsort()[:2]}")
    print(f"Least important feature: {(np.array(mean_imp)).argsort()[:2]}")

    fig, ax = plt.subplots(figsize=(10, 10))
    # plt.title("title")
    plt.xlabel('Intervention Value (alpha)', fontsize=26)
    plt.ylabel('Causal Attributions (ACE)', fontsize=26)
    im = ax.imshow(np.absolute(np.array(final)), cmap='coolwarm',
                   interpolation='nearest', norm=LogNorm())
    ax.set_xticks(np.arange(1, 21, 5))  # Since Tau was 20
    ax.set_yticks(np.arange(num_c))
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    ax.set_yticklabels(feature_names)
    ax.set_xticklabels([i for i in range(1, 21, 5)])  # Since Tau was 20
    cb = plt.colorbar(im, ax=ax)
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(20)
    plt.show()
    fig.tight_layout()
    fig.savefig(f"{filename}_mean.pdf", dpi=300)

    fig2, ax2 = plt.subplots(figsize=(10, 10))
    # plt.title("title2")
    plt.xlabel('Intervention Value (alpha)', fontsize=26)
    plt.ylabel('Causal Attributions (ACE)', fontsize=26)
    im2 = ax2.imshow(np.absolute(np.array(std)), cmap='coolwarm',
                     interpolation='nearest', norm=LogNorm())
    ax2.set_xticks(np.arange(1, 21, 5))  # Since Tau was 20
    ax2.set_yticks(np.arange(num_c))
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    ax2.set_yticklabels(feature_names)
    ax2.set_xticklabels([i for i in range(1, 21, 5)])  # Since Tau was 20
    cb2 = plt.colorbar(im2, ax=ax2)
    for t1 in cb2.ax.get_yticklabels():
        t1.set_fontsize(20)
    plt.show()
    fig2.tight_layout()
    fig2.savefig(f"{filename}_std.pdf", dpi=300)
