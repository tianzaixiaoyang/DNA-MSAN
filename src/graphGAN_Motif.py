import logging
import os

os.environ['PYTHONWARNINGS'] = 'error'
logging.getLogger().setLevel(logging.ERROR)

import tqdm
import numpy as np
import multiprocessing
import collections
import config
import utils
import pickle
import Motif_Adjacency

from scipy.stats import norm
from dfs import *
from discriminator import Discriminator
from generator import Generator

from src.EvalNE.evalne.utils import preprocess as pp
from src.EvalNE.evalne.evaluation.evaluator import LPEvaluator
from src.EvalNE.evalne.evaluation.evaluator import NCEvaluator
from src.EvalNE.evalne.evaluation.score import Scoresheet
from src.EvalNE.evalne.evaluation.split import EvalSplit
from src.EvalNE.evalne.utils import split_train_test as stt
from src.EvalNE.evalne.utils.viz_utils import *


class graphGAN(object):
    def __init__(self):
        print("reading graphs...")

        G1 = pp.load_graph(config.org_edges_filename, delimiter="\t", directed=config.directed)  
        self.all_edges_G, id_mapping = pp.prep_graph(G1)  

        self.n_node = len(self.all_edges_G.nodes)
        self.root_nodes = sorted(self.all_edges_G.nodes)

        if not os.path.exists(config.output_filename):
            os.makedirs(config.output_filename)
        pp.get_stats(G1, config.output_filename + "stats_org.txt")
        pp.get_stats(self.all_edges_G, config.output_filename + "stats_pre.txt")
        if os.path.exists(config.new_edges_filename):
            os.remove(config.new_edges_filename)
        pp.save_graph(self.all_edges_G, output_path=config.new_edges_filename, delimiter='\t', write_stats=False,
                      write_dir=False)

        all_edges = utils.read_edges(config.new_edges_filename)
        all_edges_adjacency = utils.lists_to_matrix(self.n_node, all_edges)
        all_edges_adjacency = all_edges_adjacency - np.diag(np.diag(all_edges_adjacency))

        all_edges_motifAdjacency = Motif_Adjacency.MotifAdjacency(all_edges_adjacency, "M4") + Motif_Adjacency.MotifAdjacency(all_edges_adjacency, "M13")
        if config.self_loop:
            MSAN_weighted = all_edges_motifAdjacency + np.eye(self.n_node)
        else:
            MSAN_weighted = all_edges_motifAdjacency

        self.normal_MSAN_weighted = utils.normal_matrix(MSAN_weighted)

        self.MSAN_neighbors = utils.matrix_to_lists(MSAN_weighted)

        if config.app == "link_prediction":

            self.train_test_split = EvalSplit()

            if os.path.exists(config.train_test_split + "trE_0_" + str(config.lp_train_frac) + ".csv"):
            self.train_test_split.read_splits(config.train_test_split, 0,
                                              nw_name=config.dataset,
                                              directed=config.directed,
                                              verbose=True)
            else:
                train_e, train_e_false, test_e, test_e_false = self.train_test_split.compute_splits(
                       self.all_edges_G,
                       nw_name=config.dataset,
                       train_frac=config.lp_train_frac,
                       split_alg="fast")
                stt.store_train_test_splits(config.train_test_split, train_E=train_e, train_E_false=train_e_false,
                                                test_E=test_e, test_E_false=test_e_false)

            self.lpe = LPEvaluator(self.train_test_split, dim=config.n_emb)

            self.graph = utils.read_edges(config.train_test_split + "trE_0_" + str(config.lp_train_frac) + ".csv")

        elif config.app == "node_classification":

            self.graph = utils.read_edges(config.new_edges_filename)

            self.labels = pp.read_labels(config.labels_filename, delimiter=config.delimiter, idx_mapping=id_mapping)
            self.nce = NCEvaluator(self.all_edges_G, self.labels, nw_name=config.dataset, num_shuffles=5,
                                   traintest_fracs=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], trainvalid_frac=0.6,
                                   dim=config.n_emb)
            self.scoresheet = Scoresheet()

        else:
            raise Exception("Unknown task: {}".format(config.app))

        # read pre_emb matrix
        print("generating initial embeddings for the generator...")
        node_embed_init_g = np.random.uniform(-0.1, 0.1, (self.n_node, config.n_emb))

        print("reading feature_matrix for the discriminator...")
        feature_matrix_d = utils.read_feature_matrix(filename=config.feature_matrix_filename,
                                                     n_node=self.n_node,
                                                     n_feats=config.num_feats,
                                                     old2new_idmapping=id_mapping)

        print("Initializing the generator...")
        self.generator = Generator(n_node=self.n_node, node_emd_init=node_embed_init_g)

        print("Initializing the discriminator...")
        self.discriminator = Discriminator(n_node=self.n_node, features=feature_matrix_d, adj_lists=self.MSAN_neighbors,
                                           normal_MSAN_weighted=self.normal_MSAN_weighted)
        print("Initializing the embedded vector matrix of discriminator...")
        self.discriminator.embedding_matrix = utils.get_gnn_embeddings(self.discriminator.MSAN, self.n_node)

        # construct BFS-tree
        self.trees = None
        if os.path.isfile(config.cache_filename):
            print("reading BFS-trees from cache...")
            pickle_file = open(config.cache_filename, 'rb')
            self.trees = pickle.load(pickle_file)
            pickle_file.close()
        else:
            print("constructing BFS-trees...")
            pickle_file = open(config.cache_filename, 'wb')
            if config.local_graph_softmax:
                self.trees = self.construct_trees(self.root_nodes)
            else:
                self.trees = self.construct_trees_all(self.root_nodes)
            pickle.dump(self.trees, pickle_file)
            pickle_file.close()

    def construct_trees_with_mp(self, nodes):
        """use the multiprocessing to speed up trees construction

        Args:
            nodes: the list of nodes in the graph
        """
        cores = multiprocessing.cpu_count() // 2  
        pool = multiprocessing.Pool(cores)
        new_nodes = []
        n_node_per_core = self.n_node // cores
        for i in range(cores):
            if i != cores - 1:
                new_nodes.append(nodes[i * n_node_per_core: (i + 1) * n_node_per_core])
            else:
                new_nodes.append(nodes[i * n_node_per_core:])
        trees = {}
        trees_result = pool.map(self.construct_trees, new_nodes)
        for tree in trees_result:
            trees.update(tree)
        return trees

    def construct_trees(self, nodes, d=config.BFS_depth):

        trees = {}
        for root in tqdm.tqdm(nodes):
            trees[root] = {}  
            trees[root][root] = [root]  
            used_nodes = set() 
            queue = collections.deque([root])  
            distances = {root: 0}  

            while len(queue) > 0:
                cur_node = queue.popleft() 
                max_distance = distances[cur_node]
                if max_distance > d: 
                    break
                used_nodes.add(cur_node)
                for sub_node in self.graph[cur_node]: 
                    if sub_node not in used_nodes:
                        distances[sub_node] = distances[cur_node] + 1 
                        if distances[sub_node] <= d:
                            trees[root][cur_node].append(sub_node) 
                            trees[root][sub_node] = [cur_node] 
                            queue.append(sub_node) 
                            used_nodes.add(sub_node)
        return trees

    def construct_trees_all(self, nodes):
        """use BFS algorithm to construct the BFS-trees

        Args:
            nodes: the list of nodes in the graph
        Returns:
            trees: dict, root_node_id -> tree, where tree is a dict: node_id -> list: [father, child_0, child_1, ...]
        """

        trees = {}
        for root in tqdm.tqdm(nodes):
            trees[root] = {}
            trees[root][root] = [root]
            used_nodes = set()
            queue = collections.deque([root])
            while len(queue) > 0:
                cur_node = queue.popleft()
                used_nodes.add(cur_node)
                for sub_node in self.graph[cur_node]:
                    if sub_node not in used_nodes:
                        trees[root][cur_node].append(sub_node)
                        trees[root][sub_node] = [cur_node]
                        queue.append(sub_node)
                        used_nodes.add(sub_node)
        return trees

    def prepare_data_for_d(self):
        """generate positive and negative samples for the discriminator, and record them in the txt file"""
        print("preparing data for  discriminator...")
        center_nodes = []
        neighbor_nodes = []
        labels = []
        for i in tqdm.tqdm(self.root_nodes):
            pos = list(self.graph[i])
            if len(pos) != 0:
                # positive samples
                center_nodes.extend([i] * len(pos))
                neighbor_nodes.extend(pos)
                labels.extend([1] * len(pos))

                neg_BFS, _ = self.sample(i, self.trees[i], len(pos), for_d=True)
                if neg_BFS is not None:
                    # generate negative samples by Local Graph Softmax
                    center_nodes.extend([i] * len(neg_BFS))
                    neighbor_nodes.extend(neg_BFS)
                    labels.extend([0] * len(neg_BFS))

        return center_nodes, neighbor_nodes, labels

    def prepare_data_for_g(self):
        """sample nodes for the generator"""
        print("preparing data for generator...")
        paths = []
        for i in tqdm.tqdm(self.root_nodes):
            sample, paths_from_i = self.sample(i, self.trees[i], config.n_sample_gen, for_d=False)
            if paths_from_i is not None:
                paths.extend(paths_from_i)
        node_pairs = list(map(self.get_node_pairs_from_path, paths))
        node_1 = []
        node_2 = []
        for i in range(len(node_pairs)):
            for pair in node_pairs[i]:
                node_1.append(pair[0])
                node_2.append(pair[1])
        reward = self.discriminator.reward(node_1, node_2)
        return node_1, node_2, reward

    def sample(self, root, tree, sample_num, for_d):
        """ sample nodes from BFS-tree
        Args:
            root: int, root node
            tree: dict, BFS-tree
            sample_num: the number of required samples
            for_d: bool, whether the samples are used for the generator or the discriminator
        Returns:
            samples: list, the indices of the sampled nodes
            paths: list, paths from the root to the sampled nodes
        """

        all_score = self.generator.all_score().numpy()
        samples = []
        paths = []
        n = 0

        while len(samples) < sample_num:
            current_node = root
            previous_node = -1
            paths.append([])
            is_root = True
            paths[n].append(current_node)
            while True:
                node_neighbor = tree[current_node][1:] if is_root else tree[current_node]
                is_root = False
                if len(node_neighbor) == 0:  # the tree only has a root
                    return None, None
                if for_d:  # skip 1-hop nodes (positive samples)
                    if node_neighbor == [root]:
                        # in current version, None is returned for simplicity
                        return None, None
                    if root in node_neighbor:
                        node_neighbor.remove(root)
                relevance_probability = all_score[current_node, node_neighbor] 
                relevance_probability = utils.softmax(relevance_probability)
                next_node = np.random.choice(node_neighbor, size=1, p=relevance_probability)[0]  # select next node
                paths[n].append(next_node)
                if next_node == previous_node:  # terminating condition
                    samples.append(current_node)
                    break
                previous_node = current_node
                current_node = next_node
            n = n + 1
        return samples, paths

    @staticmethod
    def get_node_pairs_from_path(path):
        """
        given a path from root to a sampled node, generate all the node pairs within the given windows size
        e.g., path = [1, 0, 2, 4, 2], window_size = 2 
        node pairs= [[1, 0], [1, 2], [0, 1], [0, 2], [0, 4], [2, 1], [2, 0], [2, 4], [4, 0], [4, 2]]
        :param path: a path from root to the sampled node
        :return pairs: a list of node pairs
        """

        path = path[:-1]
        pairs = []
        for i in range(len(path)):
            center_node = path[i]
            for j in range(max(i - config.window_size, 0), min(i + config.window_size + 1, len(path))):
                if i == j:
                    continue
                node = path[j]
                pairs.append([center_node, node])
        return pairs

    def write_embeddings_to_file(self):
        """write embeddings of the generator and the discriminator to files"""

        modes = [self.generator, self.discriminator]
        if not os.path.exists(config.results_path):
            os.makedirs(config.results_path)
        for i in range(2):
            if i == 0:
                embedding_matrix = modes[i].embedding_matrix.detach().numpy()
            else:
                embedding_matrix = modes[i].embedding_matrix
            index = np.array(range(self.n_node)).reshape(-1, 1)
            embedding_matrix = np.hstack([index, embedding_matrix])
            embedding_list = embedding_matrix.tolist()
            embedding_str = [str(int(emb[0])) + "\t" + "\t".join([str(x) for x in emb[1:]]) + "\n"
                             for emb in embedding_list]
            if not os.path.exists(config.results_path):
                os.makedirs(config.results_path)
            with open(config.emb_filenames[i], "w+") as f:
                lines = [str(self.n_node) + "\t" + str(config.n_emb) + "\n"] + embedding_str
                f.writelines(lines)

   
