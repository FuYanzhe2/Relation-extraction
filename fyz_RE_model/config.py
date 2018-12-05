import tensorflow as tf

tf.app.flags.DEFINE_string("data_path","../SemEval_data/","data root path")
tf.app.flags.DEFINE_string("embedding_file", "embedding/senna/embeddings.txt",
                           "embedding file")
tf.app.flags.DEFINE_string("embedding_vocab", "embedding/senna/words.lst",
                           "embedding vocab file")
tf.app.flags.DEFINE_string("train_file", "train.txt", "training file")
tf.app.flags.DEFINE_string("test_file", "test.txt", "Test file")
tf.app.flags.DEFINE_string("model_type", "cnn", "Test file")

tf.app.flags.DEFINE_integer("batch_size",200,"batch size")

tf.app.flags.DEFINE_integer("pos_embedding_size",5,"position embedding size")
tf.app.flags.DEFINE_integer("num_filters",500,
                            "How many features a convolution op have to output")
tf.app.flags.DEFINE_integer("class_num", 19, "Number of relations")
tf.app.flags.DEFINE_float("keep_prob", 0.5, "Dropout keep prob.")

tf.app.flags.DEFINE_integer("h_filters_windows",3,"the height of filters window ")
tf.app.flags.DEFINE_float("margin", 1, "margin based loss function")
tf.app.flags.DEFINE_float("learning_rate", 0.003, "the learning rate of trainning")
tf.app.flags.DEFINE_float("learning_decay", 1, "the learning decay rate of trainning")
tf.app.flags.DEFINE_integer("num_epoches", 200, "Number of epoches")

tf.app.flags.DEFINE_string("log_file", None, "Log file")
tf.app.flags.DEFINE_string("save_path","model/", "save model here")
tf.app.flags.DEFINE_bool('train', True, 'set True to Train')
tf.app.flags.DEFINE_bool('test', False, 'set True to test')


tf.app.flags.DEFINE_string("relations_file", "data/relations.txt", "relations file")
tf.app.flags.DEFINE_string("results_file", "data/results.txt", "predicted results file")
FLAGS=tf.app.flags.FLAGS
