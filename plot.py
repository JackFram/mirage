import matplotlib.pyplot as plt
import numpy as np
import math
import csv

#plt.rcParams['font.serif'] = "Times New Roman"
#plt.rcParams['font.family'] = "serif"
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

c1 = (31.0/255, 119/255.0, 180/255.0)
c2 = (174/255.0, 199/255.0, 232/255.0)
c3 = (1, 127/255.0, 14/255.0)
c4 = (1, 187/255.0, 120/255.0)
c5 = (44/255.0, 160/255.0, 44/255.0)
c6 = (152/255.0, 223/255.0, 138/255.0)
c7 = (214/255.0, 39/255.0, 40/255.0)
c8 = (1, 152/255.0, 150/255.0)
c9 = (148/255.0, 103/255.0, 189/255.0)
c10 = (192/255.0, 176/255.0, 213/255.0)

ablation_hybrid_speed_up = [1.14, 1.07, 1.17, 1.18, 1.15]
ablation_static_speed_up = [0.89, 0.75, 0.92, 1.06, 1.00]
ablation_sglang = [45.653, 56.378, 79.37, 114.447, 129.809]
ablation_static = [51.119, 75.195, 85.95, 107.773, 129.666]
ablation_hybrid = [39.915, 52.219, 67.414, 96.829, 112.716]

ablation_pipe_speed_up = [1.15, 1.15, 1.14, 1.28, 1.29]
ablation_pipe = [189.948, 190.551, 191.48, 194.592, 194.72]
ablation_no_pipe = [219.468, 219.967, 219.922, 249.532, 250.269]
ablation_cublas = [205.439, 205.445, 205.502, 206.381, 208.557]

# miso = results['mirage']

def plot_moe():
    def autolabel(ax, rects, num, color):
            # attach some text labels
            i = 0
            for rect in rects:
                height = ax.get_ylim()[1]
                ax.text(rect.get_x()+0.15, height, '%.2lfx' % num[i], color=color, fontsize=16, ha='center', va='bottom')
                i = i + 1
    width = 0.3
    x = [0, 1, 2, 3, 4]
    title =  "Qwen3-30B-A3B MoE Runtime"
    systems = ['SGLang-MoE', 'TGX-Static-MoE', 'TGX-Hybrid-MoE']

    fig, axes = plt.subplots(ncols = 1, nrows = 1, figsize=(8, 4), constrained_layout=True, squeeze=False)
    # fig.tight_layout()

    ax = axes[0][0]

    b0 = ax.bar(np.array(x)-1*width, ablation_sglang, width, color = c2, edgecolor="white")
    b1 = ax.bar(np.array(x)-0*width, ablation_static, width, color = c4, edgecolor = "white")
    b2 = ax.bar(np.array(x)+1*width, ablation_hybrid, width, color = c3, edgecolor = "white")
    # ax.set_xlabel(title, fontsize = 14)
    ax.set_xlim(-2.2*width, max(x) + 2.2*width)
    ax.tick_params(axis='both', which='major', labelsize=16)
    #axes[i].set_xticklabels(['A','B','C','D','E','F','G'], fontsize=12)
    autolabel(ax, b2, ablation_hybrid_speed_up, c3)

    #width2 = 0.23
    #b3 = axes[2].bar(np.array(x) - width2, np.array(ft[2]), width2, color = c10, edgecolor="white")
    #b4 = axes[2].bar(np.array(x), np.array(inc_decoding[2]), width2, color = c4, edgecolor="white")
    #b5 = axes[2].bar(np.array(x) + width2, np.array(spec_infer[2]), width2, color = c3, edgecolor="white")
    #axes[2].set_xlabel("LLaMA-65B\n(4 GPUs/node, 2 nodes)", fontsize=14)
    #axes[2].set_xlim(-2*width2, 4+2*width2)
    #axes[2].tick_params(axis='both', which='major', labelsize=12)
    #autolabel(axes[2], b5, np.array(ft[2]) / np.array(spec_infer[2]), c3)

    # print(b0)
    # print(b1)
    # print(b5)

    #plt.xticks(np.array(x) + 1.5 * width, ('GCN', 'GIN', 'GAT'))
    plt.setp(axes, xticks=[0,1,2,3,4], xticklabels=['BS=1', 'BS=2', 'BS=4', 'BS=8','BS=16'])
    fig.text(-0.032, 0.5, 'Runtime (us)', fontweight='bold', ha='left', va='center', rotation='vertical', fontsize=16)

    #fig.text(0.5, 0.02, 'Number of GPU devices', fontweight='bold', ha='center', va='bottom',  fontsize=18)
    # fig.legend([b0, b1, b2, b3, b4, b5], systems, loc = 'upper center', fontsize = 14, ncol = 6, bbox_to_anchor=(0.5,1.15))
    fig.legend([b0, b1, b2], systems, loc = 'upper center', fontsize = 16, ncol = 6, bbox_to_anchor=(0.5,1.15))

    #save to png
    plt.savefig('moe_ablation.pdf', bbox_inches='tight', dpi=100)
    # plt.show()
    
def plot_pipe():
    def autolabel(ax, rects, num, color):
            # attach some text labels
            i = 0
            for rect in rects:
                height = ax.get_ylim()[1]
                ax.text(rect.get_x()+0.15, height, '%.2lfx' % num[i], color=color, fontsize=16, ha='center', va='bottom')
                i = i + 1
    width = 0.3
    x = [0, 1, 2, 3, 4]
    title =  "Qwen3-8B LM-Head GeMM Runtime"
    systems = ['CUBLAS', 'TGX-No-Pipe', 'TGX-Pipe']

    fig, axes = plt.subplots(ncols = 1, nrows = 1, figsize=(8, 4), constrained_layout=True, squeeze=False)
    # fig.tight_layout()

    ax = axes[0][0]

    b0 = ax.bar(np.array(x)-1*width, ablation_cublas, width, color = c2, edgecolor="white")
    b1 = ax.bar(np.array(x)-0*width, ablation_no_pipe, width, color = c4, edgecolor = "white")
    b2 = ax.bar(np.array(x)+1*width, ablation_pipe, width, color = c3, edgecolor = "white")
    # ax.set_xlabel(title, fontsize = 14)
    ax.set_xlim(-2*width, max(x) + 2*width)
    ax.tick_params(axis='both', which='major', labelsize=16)
    #axes[i].set_xticklabels(['A','B','C','D','E','F','G'], fontsize=12)
    autolabel(ax, b2, ablation_pipe_speed_up, c3)

    #width2 = 0.23
    #b3 = axes[2].bar(np.array(x) - width2, np.array(ft[2]), width2, color = c10, edgecolor="white")
    #b4 = axes[2].bar(np.array(x), np.array(inc_decoding[2]), width2, color = c4, edgecolor="white")
    #b5 = axes[2].bar(np.array(x) + width2, np.array(spec_infer[2]), width2, color = c3, edgecolor="white")
    #axes[2].set_xlabel("LLaMA-65B\n(4 GPUs/node, 2 nodes)", fontsize=14)
    #axes[2].set_xlim(-2*width2, 4+2*width2)
    #axes[2].tick_params(axis='both', which='major', labelsize=12)
    #autolabel(axes[2], b5, np.array(ft[2]) / np.array(spec_infer[2]), c3)

    # print(b0)
    # print(b1)
    # print(b5)

    #plt.xticks(np.array(x) + 1.5 * width, ('GCN', 'GIN', 'GAT'))
    plt.setp(axes, xticks=[0,1,2,3,4], xticklabels=['BS=1', 'BS=2', 'BS=4', 'BS=8','BS=16'])
    fig.text(-0.032, 0.5, 'Runtime (us)', fontweight='bold', ha='left', va='center', rotation='vertical', fontsize=16)

    #fig.text(0.5, 0.02, 'Number of GPU devices', fontweight='bold', ha='center', va='bottom',  fontsize=18)
    # fig.legend([b0, b1, b2, b3, b4, b5], systems, loc = 'upper center', fontsize = 14, ncol = 6, bbox_to_anchor=(0.5,1.15))
    fig.legend([b0, b1, b2], systems, loc = 'upper center', fontsize = 16, ncol = 6, bbox_to_anchor=(0.5,1.15))

    #save to png
    plt.savefig('pipe_ablation.pdf', bbox_inches='tight', dpi=100)
    # plt.show()
    
if __name__ == "__main__":
    plot_moe()
    # plot_pipe()