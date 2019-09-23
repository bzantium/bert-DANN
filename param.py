data_root = "data"

encoder_path = "snapshots/encoder.pt"
cls_classifier_path = "snapshots/cls-classifier.pt"
dom_classifier_path = "snapshots/dom-classifier.pt"

# params for training network
num_gpu = 1
manual_seed = None

# params for optimizing models
c_learning_rate = 5e-5
d_learning_rate = 1e-5

n_vocab = 30522
hidden_size = 768
intermediate_size = 3072
class_num = 2
dropout = 0.1
num_labels = 2
