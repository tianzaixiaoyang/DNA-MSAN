import datetime
import os.path
import numpy as np
from collections import defaultdict
import re
import torch
import math
from src import config
import time
import networkx as nx


def str_list_to_float(str_list):
    return [float(item) for item in str_list]


def str_list_to_int(str_list):
    return [int(item) for item in str_list]


def read_edges_from_file(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
        edges = [str_list_to_int(re.split('[\s,]+', line)[:-1]) for line in lines]

    return edges


def read_edges(train_filename):
    adj_lists = defaultdict(set)
    train_edges = read_edges_from_file(train_filename)

    for edge in train_edges:
        if adj_lists.get(edge[0]) is None:
            adj_lists[edge[0]] = set()
        if adj_lists.get(edge[1]) is None:
            adj_lists[edge[1]] = set()
        adj_lists[edge[0]].add(edge[1])
        adj_lists[edge[1]].add(edge[0])
    return adj_lists


def get_train_nodes(graph):
    merged_data = set()
    for x in graph.keys():
        merged_data.add(x)
        merged_data.update(graph[x])
    return list(merged_data)


def read_embeddings(filename, n_node, n_embed):
    embedding_matrix = np.random.rand(n_node, n_embed)
    try:
        with open(filename, "r") as f:
            lines = f.readlines()[1:]  # skip the first line
            for line in lines:
                emd = line.split()
                embedding_matrix[int(emd[0]), :] = str_list_to_float(emd[1:])
    except Exception as e:
        print("WARNNING: can not find the pre_embedding file")
    return embedding_matrix


def read_feature_matrix(filename, n_node, n_feats, old2new_idmapping):
    feature_matrix = np.random.rand(n_node, n_feats)
    try:
        with open(filename, "r") as f:
            lines = f.readlines()
            for (oldnid, newnid) in old2new_idmapping:
                feature = lines[oldnid].split()
                feature_matrix[newnid, :] = str_list_to_float(feature[1:])
    except Exception as e:
        print("WARNNING: can not find the feature_matrix file")
    return feature_matrix


def softmax(x):
    e_x = np.exp(x - np.max(x))  # for computation stability
    return e_x / e_x.sum()


def normal_matrix(Matrix):

    sum_neighs = Matrix.sum(1)
    M = np.diag(sum_neighs)

    for x in range(M.shape[0]):
        if M[x, x] == 0:
            M[x, x] = 1

    M_inv = np.linalg.inv(M)

    R = np.matmul(M_inv, Matrix)

    return R


def get_gnn_embeddings(gnn_model, n_node):
    nodes = np.arange(n_node).tolist()
    b_sz = 500
    batches = math.ceil(len(nodes) / b_sz)

    embs = []
    for index in range(batches):
        nodes_batch = nodes[index * b_sz:(index + 1) * b_sz]
        embs_batch = gnn_model(nodes_batch)
        assert len(embs_batch) == len(nodes_batch)
        embs.append(embs_batch)

    assert len(embs) == batches
    embs = torch.cat(embs, 0)
    assert len(embs) == len(nodes)
    return embs.detach()


def adjust_learning_rate(org_lr, epoch, decay):

    return org_lr / (1 + epoch * decay)


def lists_to_matrix(n_node, adj_lists):

    MatrixAdjacency = np.zeros([n_node, n_node])
    for row in list(adj_lists.keys()):
        if len(adj_lists[row]) == 0:
            continue
        for col in list(adj_lists[row]):
            MatrixAdjacency[row, col] = 1

    return MatrixAdjacency


def matrix_to_lists(MatrixAdjacency):
    adj_lists = defaultdict(set)
    n_node = len(MatrixAdjacency)
    for i in range(n_node):
        adj_lists[i] = set()

    for row in range(n_node):
        for col in range(n_node):
            if MatrixAdjacency[row][col] != 0:
                adj_lists[row].add(col)

    return adj_lists


def load_item_pop(X_train):
    item_pop = list() 
    node_deg = dict() 
    dd = defaultdict(list)  
    for edge in X_train:
        dd[int(edge[0])].append(int(edge[1]))
        dd[int(edge[1])].append(int(edge[0]))

    for key in dd.keys():
        item_pop.append(1)
    deg_sum = np.sum(item_pop)
    for key in dd.keys():
        node_deg[key] = 1 / deg_sum

    return node_deg, dd


def construct_graph(edges):
    print("Construct the graph for training")
    G = nx.Graph()
    for edge in edges:
        x = edge[0]
        y = edge[1]
        G.add_edge(x, y)
    return G


def EvalEN(model, epoch, method_name, edge_embed_method='hadamard'):
    return_val = 0
    gen_embedding_matrix = model.generator.embedding_matrix.detach()
    index_node = np.arange(gen_embedding_matrix.shape[0]).astype(np.str).tolist()
    Xgen = dict(zip(index_node, gen_embedding_matrix))
    dis_embedding_matrix = model.discriminator.embedding_matrix
    Xdis = dict(zip(index_node, dis_embedding_matrix))
    if config.app == "link_prediction":
        results_gen = model.lpe.evaluate_ne(model.train_test_split, Xgen, method=method_name + '_den_' + str(epoch),
                                           edge_embed_method=edge_embed_method)
        results_dis = model.lpe.evaluate_ne(model.train_test_split, Xdis, method=method_name + '_dis_' + str(epoch),
                                           edge_embed_method=edge_embed_method)

        auc_gen, auc_dis = results_gen.test_scores.auroc(), results_dis.test_scores.auroc()
        fscore_gen, fscore_dis = results_gen.test_scores.f_score(), results_dis.test_scores.f_score()
        acc_gen, acc_dis = results_gen.test_scores.accuracy(), results_dis.test_scores.accuracy()

        return_val = fscore_gen

        if epoch == "pre_train":
            write_line = 'epoch: {e}\n ' \
                         '\t gen:\t auc:{auc_g:.4f}\t f_score:{fscore_g:.4f}\t acc:{acc_g:.4f}\n' \
                         '\t dis:\t auc:{auc_d:.4f}\t f_score:{fscore_d:.4f}\t acc:{acc_d:.4f}\n'.format(
                e=epoch,
                auc_g=auc_gen, fscore_g=fscore_gen, acc_g=acc_gen,
                auc_d=auc_dis, fscore_d=fscore_dis, acc_d=acc_dis)

            write_detail_gen = '\n\t auroc:{:.4f}\t f_score:{:.4f}\t acc:{:.4f}\n'.format(auc_gen, fscore_gen, acc_gen)
            write_detail_dis = '\n\t auroc:{:.4f}\t f_score:{:.4f}\t acc:{:.4f}\n'.format(auc_dis, fscore_dis, acc_dis)
        else:
            write_line = 'epoch: {e}\n ' \
                         '\t gen:\t auc:{auc_g:.4f}\t f_score:{fscore_g:.4f}\t acc:{acc_g:.4f}\n' \
                         '\t dis:\t auc:{auc_d:.4f}\t f_score:{fscore_d:.4f}\t acc:{acc_d:.4f}\n'.format(
                e=epoch,
                auc_g=auc_gen, fscore_g=fscore_gen, acc_g=acc_gen,
                auc_d=auc_dis, fscore_d=fscore_dis, acc_d=acc_dis)

            write_detail_gen = '\t auroc:{:.4f}\t f_score:{:.4f}\t acc:{:.4f}\n'.format(auc_gen, fscore_gen, acc_gen)
            write_detail_dis = '\t auroc:{:.4f}\t f_score:{:.4f}\t acc:{:.4f}\n'.format(auc_dis, fscore_dis, acc_dis)

        if not os.path.exists(config.results_path):
            os.makedirs(config.results_path)
        with open(config.results_filename, "a+") as fp:
            fp.writelines(write_line)

        for i in range(2):
            with open(config.train_detail_filename[i], "a+") as fp:
                if i == 0:
                    fp.writelines(write_detail_gen)
                else:
                    fp.writelines(write_detail_dis)

    elif config.app == "node_classification":
        results_gen = model.nce.evaluate_ne(Xgen, method_name=method_name + '_gen_' + str(epoch))
        # results_dis = model.nce.evaluate_ne(Xdis, method_name=method_name + '_dis_' + str(epoch))

        for i in results_gen:
            i.params['eval_time'] = time.time()

        model.scoresheet.log_results(results_gen) 

        f1_micro_1 = np.mean(
            model.scoresheet._scoresheet[str(config.dataset)][str(results_gen[0].method)]['f1_micro'])
        f1_micro_2 = np.mean(
            model.scoresheet._scoresheet[str(config.dataset)][str(results_gen[5].method)]['f1_micro'])
        f1_micro_3 = np.mean(
            model.scoresheet._scoresheet[str(config.dataset)][str(results_gen[10].method)]['f1_micro'])
        f1_micro_4 = np.mean(
            model.scoresheet._scoresheet[str(config.dataset)][str(results_gen[15].method)]['f1_micro'])
        f1_micro_5 = np.mean(
            model.scoresheet._scoresheet[str(config.dataset)][str(results_gen[20].method)]['f1_micro'])
        f1_micro_6 = np.mean(
            model.scoresheet._scoresheet[str(config.dataset)][str(results_gen[25].method)]['f1_micro'])
        f1_micro_7 = np.mean(
            model.scoresheet._scoresheet[str(config.dataset)][str(results_gen[30].method)]['f1_micro'])
        f1_micro_8 = np.mean(
            model.scoresheet._scoresheet[str(config.dataset)][str(results_gen[35].method)]['f1_micro'])
        f1_micro_9 = np.mean(
            model.scoresheet._scoresheet[str(config.dataset)][str(results_gen[40].method)]['f1_micro'])

        return_val = (f1_micro_1 + f1_micro_2 + f1_micro_3 + f1_micro_4 + f1_micro_5 + f1_micro_6 + f1_micro_7 + f1_micro_8 + f1_micro_9) / 9

        f1_macro_1 = np.mean(
            model.scoresheet._scoresheet[str(config.dataset)][str(results_gen[0].method)]['f1_macro'])
        f1_macro_2 = np.mean(
            model.scoresheet._scoresheet[str(config.dataset)][str(results_gen[5].method)]['f1_macro'])
        f1_macro_3 = np.mean(
            model.scoresheet._scoresheet[str(config.dataset)][str(results_gen[10].method)]['f1_macro'])
        f1_macro_4 = np.mean(
            model.scoresheet._scoresheet[str(config.dataset)][str(results_gen[15].method)]['f1_macro'])
        f1_macro_5 = np.mean(
            model.scoresheet._scoresheet[str(config.dataset)][str(results_gen[20].method)]['f1_macro'])
        f1_macro_6 = np.mean(
            model.scoresheet._scoresheet[str(config.dataset)][str(results_gen[25].method)]['f1_macro'])
        f1_macro_7 = np.mean(
            model.scoresheet._scoresheet[str(config.dataset)][str(results_gen[30].method)]['f1_macro'])
        f1_macro_8 = np.mean(
            model.scoresheet._scoresheet[str(config.dataset)][str(results_gen[35].method)]['f1_macro'])
        f1_macro_9 = np.mean(
            model.scoresheet._scoresheet[str(config.dataset)][str(results_gen[40].method)]['f1_macro'])

        with open(config.results_path + "f1_micro" + ".txt", "a+") as fp:
            if epoch == "pre_train":
                fp.writelines(
                    '\n {} \n \t f1_micro_1: {:.4f} \t f1_micro_2: {:.4f} \t f1_micro_3: {:.4f} \t f1_micro_4: {:.4f} \t f1_micro_5: {:.4f} '
                    '\t f1_micro_6: {:.4f} \t f1_micro_7: {:.4f} \t f1_micro_8: {:.4f} \t f1_micro_9: {:.4f}'.format(
                        datetime.datetime.now(), f1_micro_1, f1_micro_2, f1_micro_3, f1_micro_4, f1_micro_5,
                        f1_micro_6, f1_micro_7, f1_micro_8, f1_micro_9))
            else:
                fp.writelines(
                    '\n \t f1_micro_1: {:.4f} \t f1_micro_2: {:.4f} \t f1_micro_3: {:.4f} \t f1_micro_4: {:.4f} \t f1_micro_5: {:.4f} '
                    '\t f1_micro_6: {:.4f} \t f1_micro_7: {:.4f} \t f1_micro_8: {:.4f} \t f1_micro_9: {:.4f}'.format(
                        f1_micro_1, f1_micro_2, f1_micro_3, f1_micro_4, f1_micro_5,
                        f1_micro_6, f1_micro_7, f1_micro_8, f1_micro_9))

        with open(config.results_path + "f1_macro" + ".txt", "a+") as fp:
            if epoch == "pre_train":
                fp.writelines(
                    '\n {} \n \t f1_macro_1: {:.4f} \t f1_macro_2: {:.4f} \t f1_macro_3: {:.4f} \t f1_macro_4: {:.4f} \t f1_macro_5: {:.4f} '
                    '\t f1_macro_6: {:.4f} \t f1_macro_7: {:.4f} \t f1_macro_8: {:.4f} \t f1_macro_9: {:.4f}'.format(
                        datetime.datetime.now(), f1_macro_1, f1_macro_2, f1_macro_3, f1_macro_4, f1_macro_5,
                        f1_macro_6, f1_macro_7, f1_macro_8, f1_macro_9))
            else:
                fp.writelines(
                    '\n \t f1_macro_1: {:.4f} \t f1_macro_2: {:.4f} \t f1_macro_3: {:.4f} \t f1_macro_4: {:.4f} \t f1_macro_5: {:.4f} '
                    '\t f1_macro_6: {:.4f} \t f1_macro_7: {:.4f} \t f1_macro_8: {:.4f} \t f1_macro_9: {:.4f}'.format(
                        f1_macro_1, f1_macro_2, f1_macro_3, f1_macro_4, f1_macro_5,
                        f1_macro_6, f1_macro_7, f1_macro_8, f1_macro_9))

    else:
        raise Exception('The task {} does not exist'.format(config.app))

    return return_val

