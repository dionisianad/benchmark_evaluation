import os
import torch
import random
import inspect
import pickle as pkl
import numpy as np
from scipy import sparse
import scipy.sparse as sp
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, HeterophilousGraphDataset
from torch_geometric.utils import from_scipy_sparse_matrix, to_undirected
from torch_geometric.transforms import RandomLinkSplit, RandomNodeSplit

### Utility functions for loading and processing graph datasets

def filter_kwargs(func, kwargs):
    """Return only the kwargs that match the parameters of the given function."""
    sig = inspect.signature(func)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}

def load_data_airport(data_path, augment_features=False):
    path = os.path.join(data_path, 'airport/airport.p')
    with open(path, 'rb') as f:
        graph = pkl.load(f)

    nodes = list(graph.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    num_nodes = len(nodes)

    row, col = [], []
    for u, v in graph.edges():
        row.append(node_to_idx[u])
        col.append(node_to_idx[v])
    adj = sp.coo_matrix((np.ones(len(row)), (row, col)), shape=(num_nodes, num_nodes)).tocsr()

    features = np.array([graph.nodes[u]['feat'] for u in nodes])

    label_idx = 4
    labels = features[:, label_idx]
    features = features[:, :label_idx]
    if augment_features is True:
        deg = np.squeeze(np.sum(adj, axis=0).astype(int))
        deg = np.clip(deg, 0, 5)  # Clamp values >5 to 5
        deg_onehot = np.eye(6)[deg].squeeze()  # One-hot encode
        const_f = np.ones((features.shape[0], 1))
        features = np.concatenate((features, deg_onehot, const_f), axis=1)
    labels = np.digitize(labels, bins=[7.0 / 7, 8.0 / 7, 9.0 / 7])
    return adj, features, labels

def load_synthetic_data(dataset_str, data_dir, use_feats=True):
    object_to_idx = {}
    idx_counter = 0
    edges = []
    data_path = os.path.join(data_dir, dataset_str)
    with open(os.path.join(data_path, "{}.edges.csv".format(dataset_str)), 'r') as f:
        all_edges = f.readlines()
    for line in all_edges:
        n1, n2 = line.rstrip().split(',')
        if n1 in object_to_idx:
            i = object_to_idx[n1]
        else:
            i = idx_counter
            object_to_idx[n1] = i
            idx_counter += 1
        if n2 in object_to_idx:
            j = object_to_idx[n2]
        else:
            j = idx_counter
            object_to_idx[n2] = j
            idx_counter += 1
        edges.append((i, j))
    adj = np.zeros((len(object_to_idx), len(object_to_idx)))
    for i, j in edges:
        adj[i, j] = 1.  # comment this line for directed adjacency matrix
        adj[j, i] = 1.
    if use_feats:
        features = sp.load_npz(os.path.join(data_path, "{}.feats.npz".format(dataset_str)))
    else:
        features = sp.eye(adj.shape[0])
    labels = np.load(os.path.join(data_path, "{}.labels.npy".format(dataset_str)))
    return sp.csr_matrix(adj), features, labels


### Main class for loading graph datasets

class GraphDatasetLoader:
    def __init__(self, dataset_name, task='nc', data_dir='./data',
                 split_seed=42, *args, **kwargs):
        assert task in ['nc', 'lp']
        self.dataset_name = dataset_name.lower()
        self.task = task
        self.data_dir = data_dir
        self.split_seed = split_seed
        self.args = args
        self.all_kwargs = kwargs

        self.synthetic_datasets = ['tree1111_g00_lp', 'tree1111_g02_lp',
                              'tree1111_g04_lp', 'tree1111_g06_lp',
                              'tree1111_g08_lp', 'tree1111_g10_lp',
                              'disease_lp', 'disease_nc', 'disease_m_lp']


    def load(self):
        if self.dataset_name in ['cora', 'citeseer', 'pubmed']:
            return self._load_builtin_dataset()
        elif self.dataset_name == 'airport':
            return self._load_airport_dataset()
        elif self.dataset_name in self.synthetic_datasets:
            return self._load_custom_dataset()
        elif self.dataset_name in ["roman-empire", "amazon-ratings", "minesweeper", "tolokers", "questions"]:
            return self._load_heterophilious_dataset()
        else:
            raise ValueError(f"Dataset '{self.dataset_name}' is not supported or not recognized.")


    def _load_builtin_dataset(self):
        planetoid_kwargs = filter_kwargs(Planetoid, self.all_kwargs)
        dataset = Planetoid(root=os.path.join(self.data_dir, self.dataset_name), name=self.dataset_name.capitalize(), **planetoid_kwargs)
        data = dataset[0]

        if self.task == 'lp':
            data = self._apply_link_split(data)
        return data

    def _load_airport_dataset(self):
        adj, features, labels = load_data_airport(self.data_dir)
        edge_index, edge_attr = from_scipy_sparse_matrix(adj)
        #edge_index = to_undirected(edge_index)

        x = torch.tensor(features, dtype=torch.float)
        y = torch.tensor(labels, dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

        if self.task == 'nc':
            return self._apply_node_split(data)
        else:
            return self._apply_link_split(data)

    def _load_custom_dataset(self):
        adj, features, labels = load_synthetic_data(self.dataset_name, self.data_dir)
        edge_index, edge_attr = from_scipy_sparse_matrix(adj)
        #edge_index = to_undirected(edge_index)

        x = torch.tensor(features.todense(), dtype=torch.float)
        y = torch.tensor(labels, dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

        if self.task == 'nc':
            return self._apply_node_split(data)
        else:
            return self._apply_link_split(data)

    def _load_heterophilious_dataset(self):
        het_kwargs = filter_kwargs(HeterophilousGraphDataset, self.all_kwargs)
        dataset = HeterophilousGraphDataset(root=os.path.join(self.data_dir, self.dataset_name), name=self.dataset_name, **het_kwargs)
        data = dataset[0]
        if self.task == 'lp':
            data = self._apply_link_split(data)
        return data


    def _apply_link_split(self, data):
        link_kwargs = filter_kwargs(RandomLinkSplit, self.all_kwargs)

        # Set random seeds for reproducibility
        torch.manual_seed(self.split_seed)
        np.random.seed(self.split_seed)
        random.seed(self.split_seed)

        # remove masks and labels
        data.train_mask = data.val_mask = data.test_mask = None
        data.y = None

        # possible parameters for RandomLinkSplit
        # is_undirected = False,
        # split_labels = False,
        # num_val = 0.1,
        # num_test = 0.2,
        # add_negative_train_samples = True,
        # neg_sampling_ratio = 1.0,
        # disjoint_train_ratio = 0.0
        # edge_types = None,
        # rev_edge_types = None,
        transform = RandomLinkSplit(**link_kwargs)
        return transform(data)

    def _apply_node_split(self, data):
        if 'ratio_val' in self.all_kwargs:
            ratio_val = self.all_kwargs.get('ratio_val', 0.1)
            test_val = self.all_kwargs.get('test_val', 0.1)
            num_val = int(data.num_nodes * ratio_val)
            num_test = int(data.num_nodes * test_val)
            node_kwargs = filter_kwargs(RandomLinkSplit, self.all_kwargs)
            node_kwargs['num_val'] = num_val
            node_kwargs['num_test'] = num_test
        else:
            node_kwargs = filter_kwargs(RandomNodeSplit, self.all_kwargs)
        # Set random seeds for reproducibility
        torch.manual_seed(self.split_seed)
        np.random.seed(self.split_seed)
        random.seed(self.split_seed)

        transform = RandomNodeSplit(**node_kwargs)
        return transform(data)

if __name__ == '__main__':
    # # Example usage on Planetoid datasets
    print("Loading Cora dataset for node classification...")
    task = 'nc'  # 'nc' for node classification, 'lp' for link prediction
    loader_nc = GraphDatasetLoader('cora', task=task, split='geom-gcn')
    data_nc = loader_nc.load()
    print(data_nc.x[data_nc.train_mask[:, 0].bool()].shape)
    print(data_nc.x[data_nc.val_mask[:, 0].bool()].shape)
    print(data_nc.x[data_nc.test_mask[:, 0].bool()].shape)

    print("Loading Cora dataset for link prediction...")
    task ='lp'
    loader = GraphDatasetLoader('cora', task=task, num_val=0.1, num_test=0.2, split_labels=True, is_undirected=True)
    train_data, val_data, test_data = loader.load()
    print(train_data)
    print(val_data)
    print(test_data)

    # Example on Heterophilous datasets
    print("Loading Roman Empire dataset for node classification...")
    task = 'nc'
    loader_nc = GraphDatasetLoader('roman-empire', task=task)
    data_nc = loader_nc.load()
    print(data_nc.x[data_nc.train_mask[:, 0]].shape)
    print(data_nc.x[data_nc.val_mask[:, 0]].shape)
    print(data_nc.x[data_nc.test_mask[:, 0]].shape)

    print("Loading Roman Empire dataset for link prediction...")
    task ='lp'
    loader = GraphDatasetLoader('roman-empire', task=task, num_val=0.1, num_test=0.2, split_labels=True, is_undirected=False)
    train_data, val_data, test_data = loader.load()
    print(train_data)
    print(val_data)
    print(test_data)

    # Example on Airport, Disease, Tree1111 and other custom datasets
    print("Loading Airport dataset for node classification...")
    task= 'nc'
    loader_nc = GraphDatasetLoader('airport', task=task, ratio_val=0.1, test_val=0.6, split='train_rest')
    data_nc=loader_nc.load()
    print(data_nc.x[data_nc.train_mask].shape)
    print(data_nc.x[data_nc.val_mask].shape)
    print(data_nc.x[data_nc.test_mask].shape)

    print("Loading Airport dataset for link prediction...")
    task = 'lp'
    loader = GraphDatasetLoader('airport', task=task, num_val=0.1, num_test=0.2, split_labels=True, is_undirected=True)
    train_data, val_data, test_data = loader.load()
    print(train_data)
    print(val_data)
    print(test_data)



    # # Example usage
    # task= 'nc'  # 'nc' for node classification, 'lp' for link prediction
    # loader_nc = GraphDatasetLoader('airport', task=task, ratio_val=0.1, test_val=0.6, split='train_rest')
    # data_nc=loader_nc.load()
    # print(data_nc.x[data_nc.train_mask].shape)
    # print(data_nc.x[data_nc.val_mask].shape)
    # print(data_nc.x[data_nc.test_mask].shape)
    #