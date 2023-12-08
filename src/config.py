modes = ["gen", "dis"]

# training settings
batch_size_gen = 64  # batch size for the generator
batch_size_dis = 64  # batch size for the discriminator
lambda_gen = 1e-5  # l2 loss regulation weight for the generator
lambda_dis = 1e-5  # l2 loss regulation weight for the discriminator
n_sample_gen = 5  # 20 number of samples for the generator

lr_gen = 3e-5  # learning rate for the generator
lr_dis = 1e-5  # learning rate for the discriminator

n_epochs = 100  # number of outer loops
n_epochs_gen = 30  # number of inner loops for the generator
n_epochs_dis = 30  # number of inner loops for the discriminator

# other hyper-parameters
n_emb = 128
multi_processing = False  # whether using multi-processing to construct BFS-trees
window_size = 2  

# application and dataset settings
app = ["link_prediction", "node_classification"][0]
# select the dataset
dataset_num = 3
dataset = ["lastfm", "citeseer", "cora", "wiki"][dataset_num]
num_feats = [7842, 3703, 1433, 2405][dataset_num]
directed = [False, False, False, True][dataset_num]
delimiter = ["\t", "\t", "\t", " "][dataset_num]

lp_train_frac = 0.7
self_loop = False
local_graph_softmax = True
BFS_depth = 2
# project path
project_path = "/home/teach/ying/DNA-MSAN/"

org_edges_filename = project_path + "data/" + dataset + "/" + dataset + ".cites"
org_feature_filename = project_path + "data/" + dataset + "/" + dataset + ".content"
org_labels_filename = project_path + "data/" + dataset + "/" + dataset + ".labels"

output_filename = project_path + "data/" + dataset + "/output/"
new_edges_filename = output_filename + dataset + "_pre.cites"

lp_train_filename = project_path + "data/" + dataset + "/train_test_split/" + "trE_0.csv"
lp_test_filename = project_path + "data/" + dataset + "/train_test_split/" + "teE_0.csv"
lp_test_neg_filename = project_path + "data/" + dataset + "/train_test_split/" + "negTeE_0.csv"
labels_filename = project_path + "data/" + dataset + "/" + dataset + ".labels"

if app == "link_prediction":
    if local_graph_softmax:
        cache_filename = project_path + "cache/" + app + "/" + dataset + "_lgs_" + str(lp_train_frac) + ".pkl"
    else:
        cache_filename = project_path + "cache/" + app + "/" + dataset + "_gs_" + str(lp_train_frac) + ".pkl"
else:
    if local_graph_softmax:
        cache_filename = project_path + "cache/" + app + "/" + dataset + "_lgs" + ".pkl"
    else:
        cache_filename = project_path + "cache/" + app + "/" + dataset + "_gs" + ".pkl"

train_test_split = project_path + "data/" + dataset + "/train_test_split/"

feature_matrix_filename = project_path + "data/" + dataset + "/" + dataset + ".content"

results_path = project_path + "results/" + app + "/" + dataset + "/"
results_filename = results_path + dataset + ".txt"
train_detail_filename = [results_path + "gen_detail_" + ".txt",
                         results_path + "dis_detail_" + ".txt"]
emb_filenames = [results_path + dataset + "_gen_" + ".emb",
                 results_path + dataset + "_dis_" + ".emb"]

time_filename = results_path + "epoch_time.txt"
