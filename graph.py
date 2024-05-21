import torch
from torch import nn
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

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

    matrix = np.array([
        [-1.88, -0.47, -0.41, -1.94, -2.40,  2.12],
        [ 0.00,  2.15,  1.89, -1.83, -1.46,  0.77],
        [ 0.00,  0.00,  0.65, -2.04, -2.12,  0.74],
        [ 0.00,  0.00,  0.00, -2.12, -1.59,  1.46],
        [ 0.00,  0.00,  0.00,  0.00, -1.61,  1.77],
        [ 0.00,  0.00,  0.00,  0.00,  0.00,  2.46],
    ])
    matrix = matrix + matrix.T

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

def draw():
    matrix = np.array([
        [-1.88, -0.47, -0.41, -1.94, -2.40,  2.12],
        [ 0.00,  2.15,  1.89, -1.83, -1.46,  0.77],
        [ 0.00,  0.00,  0.65, -2.04, -2.12,  0.74],
        [ 0.00,  0.00,  0.00, -2.12, -1.59,  1.46],
        [ 0.00,  0.00,  0.00,  0.00, -1.61,  1.77],
        [ 0.00,  0.00,  0.00,  0.00,  0.00,  2.46],
    ])

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
        [0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
    ])
    p = torch.tensor([1, 2, 3, 4, 5, 6])

    print(torch.matmul(p, m.t()))
    print((p[None, :] * m).sum(1))

if __name__ == '__main__':
    mat()
