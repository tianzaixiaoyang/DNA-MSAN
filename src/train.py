import torch
import numpy as np
import config
import utils
import random
import time
from graphGAN_Motif import graphGAN
from EvalNE.evalne.utils.viz_utils import *


def show_progress(item_name, epoch, batch_index, batch_num, loss):

    barlen = 40
    ok_len = int((batch_index + 1) / batch_num * barlen)

    epoch_info = 'epoch:{}'.format(epoch + 1)
    loss_info = 'loss: {}'.format(loss)
    batch_info = '{}/{}'.format(batch_index + 1, batch_num)
    bar_str = '[' + '>' * ok_len + '-' * (barlen - ok_len) + ']'
    info_end = '\r' 
    info_list = [item_name, epoch_info, batch_info, bar_str, loss_info]

    if batch_index + 1 == batch_num:
        info_end = '\n' 

    progress_info = ' '.join(info_list)
    print(progress_info, end=info_end, flush=True)


def D_step(gGAN, optmizer):
    optimizer_D = optmizer

    center_nodes, neighbor_nodes, labels = gGAN.prepare_data_for_d()
    print("Data preparation is completed, start training the discriminator...")
    for d_epoch in range(config.n_epochs_dis):
        train_size = len(center_nodes)
        start_list = list(range(0, train_size, config.batch_size_dis))
        np.random.shuffle(start_list)

        all_loss = 0
        for i, start in enumerate(start_list):
            end = start + config.batch_size_dis

            loss = gGAN.discriminator.loss(node_id=center_nodes[start:end],
                                           node_neighbor_id=neighbor_nodes[start:end],
                                           label=labels[start:end])

            optimizer_D.zero_grad()
            loss.backward()
            optimizer_D.step()

            all_loss = all_loss + loss.detach().numpy()
            avg_loss = all_loss / (i + 1)
            show_progress('dis', d_epoch, i, len(start_list), avg_loss)
    gGAN.discriminator.embedding_matrix = utils.get_gnn_embeddings(gGAN.discriminator.CAN, gGAN.n_node)


def G_step(gGAN, optmizer):
    optimizer_G = optmizer

    node_1, node_2, reward = gGAN.prepare_data_for_g()
    print("Data preparation is completed, start training the generator...")

    for g_epoch in range(config.n_epochs_gen):
        train_size = len(node_1)
        start_list = list(range(0, train_size, config.batch_size_gen))
        np.random.shuffle(start_list)

        all_loss = 0
        for i, start in enumerate(start_list):
            end = start + config.batch_size_gen
            score = gGAN.generator.score(node_id=np.array(node_1[start:end]),
                                         node_neighbor_id=np.array(node_2[start:end]))

            prob = torch.sigmoid(score)
            loss = gGAN.generator.loss(prob=prob,
                                       reward=reward[start:end])

            optimizer_G.zero_grad()
            loss.backward()
            optimizer_G.step()

            all_loss = all_loss + loss.detach().numpy()
            avg_loss = all_loss / (i + 1)
            show_progress('gen', g_epoch, i, len(start_list), avg_loss)


def train():
    gGAN = graphGAN()

    optimizer_D = torch.optim.Adam(gGAN.discriminator.parameters())
    optimizer_G = torch.optim.Adam(gGAN.generator.parameters())

    gGAN.write_embeddings_to_file()
    max_val = utils.EvalEN(gGAN, epoch="pre_train", method_name="DNA-MSAN")

    patience = 10
    count = 0
    all_times = []

    print("start training...")
    for epoch in range(config.n_epochs):
        print(" epoch %d " % epoch)
        start_time = time.time()

        # D-steps
        optimizer_D.param_groups[0]["lr"] = utils.adjust_learning_rate(org_lr=config.lr_dis,
                                                                       epoch=epoch,
                                                                       decay=0.01)
        D_step(gGAN, optimizer_D)

        # G-steps
        optimizer_G.param_groups[0]["lr"] = utils.adjust_learning_rate(org_lr=config.lr_gen,
                                                                       epoch=epoch,
                                                                       decay=0.06)

        G_step(gGAN, optimizer_G)

        end_time = time.time()
        spend_time = (end_time - start_time) / 60
        all_times.append(spend_time)

        write_line = '\t epoch: {}\t spend_time: {:.2f}  mins\n'.format(epoch, spend_time)
        with open(config.time_filename, "a+") as f:
            f.writelines(write_line)

        x = utils.EvalEN(gGAN, epoch=epoch, method_name="DNA-MSAN")
        if x > max_val:
            max_val = x
            gGAN.write_embeddings_to_file()
            count = 0
        else:
            count += 1

        if count == patience:
            write_line = '\n\t mean_times: {:.2f}  mins\n'.format(np.mean(all_times))
            with open(config.time_filename, "a+") as f:
                f.writelines(write_line)
            break

    print("training completes")


if __name__ == "__main__":
    # link prediction：       cora:1004; citeseer:1005; wiki:1006; LastFM:1007
    # node classification：cora:2004; citeseer:2005; wiki:2006; LastFM:2007
    print("Designing random number seed...")
    seed = 1006
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    train()
