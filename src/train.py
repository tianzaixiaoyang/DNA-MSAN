import torch
import numpy as np
import config
import utils
from model import DNA_MSAN


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


def D_step(model, optmizer):
    optimizer_D = optmizer

    center_nodes, neighbor_nodes, labels = model.prepare_data_for_d()
    print("Data preparation is completed, start training the discriminator...")
    for d_epoch in range(config.n_epochs_dis):
        train_size = len(center_nodes)
        start_list = list(range(0, train_size, config.batch_size_dis))
        np.random.shuffle(start_list)

        all_loss = 0
        for i, start in enumerate(start_list):
            end = start + config.batch_size_dis

            loss = model.discriminator.loss(node_id=center_nodes[start:end],
                                           node_neighbor_id=neighbor_nodes[start:end],
                                           label=labels[start:end])

            optimizer_D.zero_grad()
            loss.backward()
            optimizer_D.step()

            all_loss = all_loss + loss.detach().numpy()
            avg_loss = all_loss / (i + 1)
            show_progress('dis', d_epoch, i, len(start_list), avg_loss)
    model.discriminator.embedding_matrix = utils.get_gnn_embeddings(model.discriminator.MSAN, model.n_node)


def G_step(model, optmizer):
    optimizer_G = optmizer

    node_1, node_2, reward = model.prepare_data_for_g()
    print("Data preparation is completed, start training the generator...")

    for g_epoch in range(config.n_epochs_gen):
        train_size = len(node_1)
        start_list = list(range(0, train_size, config.batch_size_gen))
        np.random.shuffle(start_list)

        all_loss = 0
        for i, start in enumerate(start_list):
            end = start + config.batch_size_gen
            score = model.generator.score(node_id=np.array(node_1[start:end]),
                                         node_neighbor_id=np.array(node_2[start:end]))

            prob = torch.sigmoid(score)
            loss = model.generator.loss(prob=prob, reward=reward[start:end])

            optimizer_G.zero_grad()
            loss.backward()
            optimizer_G.step()

            all_loss = all_loss + loss.detach().numpy()
            avg_loss = all_loss / (i + 1)
            show_progress('gen', g_epoch, i, len(start_list), avg_loss)


def train():
    model = DNA_MSAN()

    optimizer_D = torch.optim.Adam(model.discriminator.parameters())
    optimizer_G = torch.optim.Adam(model.generator.parameters())

    max_val = utils.EvalEN(model, epoch="pre_train", method_name="DNA-MSAN")

    patience = 10
    count = 0

    print("start training...")
    for epoch in range(config.n_epochs):
        print(" epoch %d " % epoch)

        # D-steps
        D_step(model, optimizer_D)

        # G-steps
        G_step(model, optimizer_G)

        x = utils.EvalEN(model, epoch=epoch, method_name="DNA-MSAN")
        if x > max_val:
            max_val = x
            count = 0
        else:
            count += 1

        if count == patience:
            break

    print("training completes")


if __name__ == "__main__":
    train()
