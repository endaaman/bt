import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform

from endaaman import BaseCLI


class Graph(nn.Module):
    def __init__(self, size, depth):
        super().__init__()
        self.size = size
        self.depth = depth
        self.mask = torch.triu(torch.ones(size, size), diagonal=1)

        # torch.empty(depth, size, size)
        # nn.init.normal_(param, mean=0, std=1)
        param = torch.normal(mean=0, std=1, size=(size, size))
        param[self.mask<1] = 0.0
        self.param = nn.Parameter(param)

    def get_matrix(self):
        return (self.param * self.mask)

    def forward(self, preds):
        l = (preds * self.get_matrix()).sum(dim=1).log()



def graph():
    g = Graph(5, 3)
    print('param', g.param)
    print('mask', g.mask)
    print('masked', g.masked_param())
    print('masked', torch.softmax(g.masked_param(), dim=-1))
    # print(torch.softmax(g.param, dim=-1))

def mat():
    m = torch.tensor([
        [1000, 0, 0, 0, 0],
        [0, 1000, 0, 0, 0],
        [0, 0, 333, 666, 666],
        [0, 0, 0, 333, 666],
        [0, 0, 0, 0, 333],

        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0.5, 0, 0],
        [0, 0, 0, 0.5, 0.5, 0],
        [0, 0, 0, 0, 0.5 ],
    ]).float()

    m = (m + m.t())/2
    print(m.softmax(dim=0))

    q  = torch.tensor([[0.01, 0.96, 0.01, 0.01, 0.01]])
    p  = torch.tensor([[0.01, 0.01, 0.96, 0.01, 0.01]])
    r  = torch.tensor([[0.01, 0.01, 0.01, 0.01, 0.96]])
    gt = torch.tensor([[0, 0, 1, 0, 0]])

    p2= (p*m).sum(dim=1).log()
    q2= (q*m).sum(dim=1).log()
    r2= (r*m).sum(dim=1).log()
    gt2= (gt*m).sum(dim=1)
    print(gt2)

    print((p2*gt2).sum())
    print((q2*gt2).sum())
    print((r2*gt2).sum())


def draw():
    # c = torch.load('out/nested/LMGAOB_graph_gamma2_dim0/checkpoint_last.pt')
    c = torch.load('out/nested/LMGAOB_graph_gamma2_dim1/checkpoint_last.pt')
    # c = torch.load('out/nested/LMGAOB_graph_gamma2_dimx/checkpoint_last.pt')
    # c = torch.load('out/nested/LMGAO__graph_dim0/checkpoint_last.pt')

    matrix = c['model_state']['graph_matrix.matrix'].numpy()
    # matrix=(matrix-matrix.min())/(matrix.max()-matrix.min())

    G = nx.Graph()

    # 行列からエッジと重みを追加
    rows, cols = np.where(matrix > 0)
    edges = zip(rows.tolist(), cols.tolist())
    G.add_edges_from(edges)

    # エッジの重みを設定
    for i, j in edges:
        G[i][j]['weight'] = matrix[i, j]

    # グラフを描画
    pos = nx.spring_layout(G)  # グラフのレイアウトを決定
    nx.draw(G, pos, with_labels=True, font_weight='bold')
    labels = dict(enumerate(['L', 'M', 'GBM', 'A', 'O', 'B'][:matrix.shape[0]]))
    labels = nx.get_edge_attributes(G, 'weight')
    print(labels)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

    plt.axis('off')
    plt.show()

def draw_with_weight():
    # matrix = np.array([
    #     [-0.18, -0.41, -0.40, -0.17,  0.04,  0.32],
    #     [ 0.00, -0.23, -0.44, -0.21,  0.01,  0.28],
    #     [ 0.00,  0.00, -0.21, -0.20,  0.02,  0.29],
    #     [ 0.00,  0.00,  0.00,  0.02,  0.24,  0.54],
    #     [ 0.00,  0.00,  0.00,  0.00,  0.22,  0.74],
    #     [ 0.00,  0.00,  0.00,  0.00,  0.00,  0.54],
    # ])

    # matrix = np.array([
    #     [-2.49, -4.32, -4.34, -4.00, -4.59,  7.06],
    #     [ 0.00, -2.02, -4.09, -3.69, -4.17,  7.31],
    #     [ 0.00,  0.00, -2.11, -3.76, -4.23,  7.29],
    #     [ 0.00,  0.00,  0.00, -1.81, -4.11,  7.62],
    #     [ 0.00,  0.00,  0.00,  0.00, -2.48,  7.09],
    #     [ 0.00,  0.00,  0.00,  0.00,  0.00,  9.32],
    # ])

    # matrix = np.array([
    #     [-1.88, -0.47, -0.41, -1.94, -2.40,  2.12],
    #     [ 0.00,  2.15,  1.89, -1.83, -1.46,  0.77],
    #     [ 0.00,  0.00,  0.65, -2.04, -2.12,  0.74],
    #     [ 0.00,  0.00,  0.00, -2.12, -1.59,  1.46],
    #     [ 0.00,  0.00,  0.00,  0.00, -1.61,  1.77],
    #     [ 0.00,  0.00,  0.00,  0.00,  0.00,  2.46],
    # ])
    # matrix = matrix + matrix.T

    c = torch.load('out/nested/LMGAOB_graph_gamma2_dim1/checkpoint_last.pt')
    matrix = c['model_state']['graph_matrix.matrix'].numpy()

    # NetworkXのグラフオブジェクトを作成
    G = nx.Graph()

    # 行列からエッジと重みを追加
    rows, cols = np.where(matrix != 0)
    edges = zip(rows.tolist(), cols.tolist())
    G.add_weighted_edges_from([(i, j, abs(matrix[i, j])) for i, j in edges])

    # 自己ループを追加
    for i in range(matrix.shape[0]):
        if matrix[i, i] != 0:
            G.add_edge(i, i, weight=abs(matrix[i, i]))

    # エッジの太さと色を重みに応じて設定
    edge_width = [abs(G[u][v]['weight']) * 5 for u, v in G.edges()]
    edge_color = ['r' if G[u][v]['weight'] < 0 else 'b' for u, v in G.edges()]

    # グラフを描画
    pos = nx.spring_layout(G)  # グラフのレイアウトを決定
    nx.draw(G, pos, with_labels=True, font_weight='bold', width=edge_width, edge_color=edge_color)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

    # 自己ループのエッジを調整
    self_loops = list(nx.selfloop_edges(G))
    nx.draw_networkx_edges(G, pos, edgelist=self_loops, width=[abs(G[u][v]['weight']) * 5 for u, v in self_loops], edge_color=['r' if G[u][v]['weight'] < 0 else 'b' for u, v in self_loops], arrows=True, arrowsize=20)

    plt.axis('off')
    plt.show()

def draw_all():
    # matrix = np.array([
    #     [-1.88, -0.47, -0.41, -1.94, -2.40,  2.12],
    #     [ 0.00,  2.15,  1.89, -1.83, -1.46,  0.77],
    #     [ 0.00,  0.00,  0.65, -2.04, -2.12,  0.74],
    #     [ 0.00,  0.00,  0.00, -2.12, -1.59,  1.46],
    #     [ 0.00,  0.00,  0.00,  0.00, -1.61,  1.77],
    #     [ 0.00,  0.00,  0.00,  0.00,  0.00,  2.46],
    # ])
    c = torch.load('out/nested/LMGAOB_graph_gamma2_dim0/checkpoint_last.pt')
    matrix = c['model_state']['graph_matrix.matrix'].numpy()

    G = nx.DiGraph()

    rows, cols = np.triu_indices_from(matrix, k=1)
    edges = zip(rows.tolist(), cols.tolist())
    G.add_weighted_edges_from([(i, j, matrix[i, j]) for i, j in edges])

    # 自己ループを追加
    for i in range(matrix.shape[0]):
        G.add_edge(i, i, weight=matrix[i, i])

    # グラフを描画
    pos = nx.spring_layout(G)  # グラフのレイアウトを決定
    nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=1000, node_color='lightblue', edge_color='gray', arrows=True, connectionstyle='arc3,rad=0.1')
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=10, label_pos=0.3)

    plt.axis('off')
    plt.show()


def mat():
    m = torch.tensor([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1],
    ])
    p = torch.tensor([0, 1, 2, 3, 4, 5])
    p = F.one_hot(p)


    # print(torch.matmul(p, m.t()))
    # print((p[None, :] * m).sum(1))


class CLI(BaseCLI):
    class DendroArgs(BaseCLI.CommonArgs):
        load: bool = False

    def run_dendro(self, a):
        torch.set_printoptions(precision=2, sci_mode=False)

        labels = None
        mat = np.array([
            [1.0, 0.0],  # 0
            [0.6, 0.4],  # 1
            [0.5, 0.5],  # 5
            [0.2, 0.8],  # 4
            [0.2, 0.8],  # 2
            [0.1, 0.9],  # 3
        ])

        if a.load:
            labels = list('LMGAOB')
            c = torch.load('out/nested/LMGAOB_graph_gamma2_dim0/checkpoint_last.pt')
            # c = torch.load('out/nested/LMGAOB_graph_dual/checkpoint_last.pt')
            mat = c['model_state']['graph_matrix.matrix']
            mat = (mat+mat.t())/2
            # mat = torch.softmax(mat, dim=0)
            # mat = torch.sigmoid(mat)

        distant_matrix = pdist(mat, metric='euclidean')
        linkage_matrix = linkage(distant_matrix, method='ward')

        # distant_matrix = np.corrcoef(mat)
        # linkage_matrix = linkage(distant_matrix, method='ward')

        plt.figure(figsize=(10, 5))
        dendrogram(linkage_matrix, labels=labels)
        plt.xlabel('Samples')
        plt.ylabel('Distance')
        plt.title('Dendrogram')
        plt.tight_layout()
        plt.show()

    def run_dendro2(self, a):
        torch.set_printoptions(precision=2, sci_mode=False)

        labels = list('LMGAOB')
        c = torch.load('out/nested/LMGAOB_hier/checkpoint_last.pt')
        mat6x2 = c['model_state']['hier_matrixes.matrixes.0']
        mat6x3 = c['model_state']['hier_matrixes.matrixes.1']
        mat6x4 = c['model_state']['hier_matrixes.matrixes.2']
        mat6x5 = c['model_state']['hier_matrixes.matrixes.3']

        print(mat6x2)
        print(mat6x3)

        combined_matrix = np.hstack((mat6x2, mat6x3, mat6x4, mat6x5))
        distant_matrix = pdist(combined_matrix, metric='euclidean')
        linkage_matrix = linkage(distant_matrix, method='ward')

        plt.figure(figsize=(10, 5))
        dendrogram(linkage_matrix, labels=labels)
        plt.xlabel('Samples')
        plt.ylabel('Distance')
        plt.title('Dendrogram')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    cli = CLI()
    cli.run()
