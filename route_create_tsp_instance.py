import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import itertools


def combination_num(n, k):
    """
    :param n:样本数
    :param k: 抽取数
    :return: 组合数
    """

    def factorial(n):
        """
        :param n: 输入
        :return: 阶乘
        """
        if n == 1 or n == 0:
            return 1
        else:
            return n * factorial(n - 1)

    return factorial(n) // (factorial(n - k) * factorial(k))


def creat_data(nodes_origin, n_nodes=20, iter_per_batch=100, batch_n=100):
    # nodes 的第一个点将被始终保留作为起点
    def c_dist(x1, x2):
        return ((x1[0] - x2[0]) ** 2 + (x1[1] - x2[1]) ** 2) ** 0.5

    def cal_edges(data_node, date_n):
        edges = np.zeros((date_n, date_n, 1))
        for i, (x1, y1) in enumerate(data_node):
            for j, (x2, y2) in enumerate(data_node):
                d = c_dist((x1, y1), (x2, y2))
                edges[i][j][0] = d

        edges = edges.reshape(-1, 1)
        return edges

    nodes = nodes_origin[1:]
    edges_index = []
    for i in range(n_nodes):  # n_nodes包含起始点
        for j in range(n_nodes):
            edges_index.append([i, j])
    edges_index = torch.LongTensor(edges_index)
    edges_index = edges_index.transpose(dim0=0, dim1=1)

    datas = []
    batch_size_per = combination_num(len(nodes), n_nodes - 1)
    batch_size = (iter_per_batch // batch_size_per) * batch_size_per

    for i in range(batch_n):
        for k in range(iter_per_batch // batch_size_per):
            for j in itertools.combinations(nodes, n_nodes - 1):
                node = np.array([nodes_origin[0]] + list(j))
                edge = cal_edges(node, n_nodes)
                data = Data(x=torch.from_numpy(node).float(), edge_index=edges_index,
                            edge_attr=torch.from_numpy(edge).float())
                datas.append(data)
    # print(datas)
    dl = DataLoader(datas, batch_size=batch_size)
    return dl, batch_size


# nodes(batch_size,2,n_nodes)
# edges(batch_size,n_nodes,n_nodes,1)
# edges_index(batch_size,2,n_nodes)
# dynamic(batch_size,1,n_nodes)
def reward(static, tour_indices, n_nodes, batch_size):
    static = static.reshape(-1, n_nodes, 2)
    # print(static.shape)
    static = static.transpose(2, 1)
    tour_indices = tour_indices.reshape(batch_size, n_nodes)
    idx = tour_indices.unsqueeze(1).expand_as(static)
    tour = torch.gather(static.data, 2, idx).permute(0, 2, 1)
    # print(tour.shape)
    # print(idx.shape)
    y = torch.cat((tour, tour[:, :1]), dim=1)

    tour_len = torch.sqrt(torch.sum(torch.pow(y[:, :-1] - y[:, 1:], 2), dim=2))
    # print(tour_len.sum(1))
    return tour_len.sum(1).detach()


def reward1(static, tour_indices, n_nodes):
    # static = static.transpose(2,1)
    # print(static.shape)  static(batch_size*n_nodes,2)
    # print(static.shape,static)
    static = static.reshape(-1, n_nodes, 2)
    # print(static.shape,static)
    static = static.transpose(2, 1)
    idx = tour_indices.unsqueeze(1).expand_as(static)
    tour = torch.gather(static, 2, idx).permute(0, 2, 1)
    # print(tour.shape,tour[0])
    # print(idx.shape,idx[0])
    # Make a full tour by returning to the start
    y = torch.cat((tour, tour[:, :1]), dim=1)

    # Euclidean distance between each consecutive point
    tour_len = torch.sqrt(torch.sum(torch.pow(y[:, :-1] - y[:, 1:], 2), dim=2))
    # print(tour_len.sum(1))
    return tour_len.sum(1).detach()
