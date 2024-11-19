modes = ["gen", "dis"]

# training settings
batch_size_gen = 64  # batch size for the generator
batch_size_dis = 64  # batch size for the discriminator
lambda_gen = 1e-5  # l2 loss regulation weight for the generator
lambda_dis = 1e-5  # l2 loss regulation weight for the discriminator
n_sample_gen = 5  # 5 number of samples for the generator

lr_gen = 3e-5  # learning rate for the generator
lr_dis = 1e-5  # learning rate for the discriminator

n_epochs = 100  # number of outer loops
n_epochs_gen = 30  # number of inner loops for the generator
n_epochs_dis = 30  # number of inner loops for the discriminator

# other hyper-parameters
n_emb = 128
window_size = 2  

# application and dataset settings
app = ["link_prediction", "node_classification"][1]
# select the dataset
dataset_num = 1
dataset = ["lastfm", "citeseer", "cora", "wiki"][dataset_num]
num_feats = [7842, 3703, 1433, 2405][dataset_num]
directed = [False, False, False, True][dataset_num]
delimiter = ["\t", "\t", "\t", " "][dataset_num]

lp_train_frac = 0.7  # The training ratio for link prediction

# The depth of the local graph softmax
BFS_depth = 2

# project path
project_path = "home/teach/DNA-MSAN/"

# The paths to the original edge set, attribute matrix, and labels.
org_edges_filename = project_path + "data/" + dataset + "/" + dataset + ".cites"
org_feature_filename = project_path + "data/" + dataset + "/" + dataset + ".content"
org_labels_filename = project_path + "data/" + dataset + "/" + dataset + ".labels"

# The paths to the edge set and attribute matrix after preprocessing with EvalNE.
output_filename = project_path + "data/" + dataset + "/output/"
new_edges_filename = output_filename + dataset + "_pre.cites"

lp_train_filename = project_path + "data/" + dataset + "/train_test_split/" + "trE_0.csv"
lp_test_filename = project_path + "data/" + dataset + "/train_test_split/" + "teE_0.csv"
lp_test_neg_filename = project_path + "data/" + dataset + "/train_test_split/" + "negTeE_0.csv"
labels_filename = project_path + "data/" + dataset + "/" + dataset + ".labels"

train_test_split = project_path + "data/" + dataset + "/train_test_split/"

feature_matrix_filename = project_path + "data/" + dataset + "/" + dataset + ".content"

results_path = project_path + "results/" + app + "/" + dataset + "/"
results_filename = results_path + dataset + ".txt"
