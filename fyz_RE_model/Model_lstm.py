import tensorflow as tf
import numpy as np
from sklearn import metrics
from tensorflow.contrib import rnn
from tensorflow.contrib.layers.python.layers import initializers
class Model(object):
    def __init__(self,config,is_Training=True):
        self.config = config
        self.is_Training = is_Training
        self.batch_size = config.batch_size
        self.max_len = config.max_length
        self.r_class = config.class_num  # number of relations
        self.num_filters = config.num_filters
        self.input_feature = config.sentence_vec_dim+self.config.pos_embedding_size*2
        self.dropout = config.keep_prob
        self.h_filters_windows=config.h_filters_windows
        self.margin = config.margin
        lr = config.learning_rate
        decay = config.learning_decay

        with tf.name_scope("input_layer"):
            self.input_sentences = tf.placeholder(dtype=tf.int32,shape=[self.batch_size,self.max_len],
                                                  name="input_S")

            self.distant_1 = tf.placeholder(dtype=tf.int32,shape=[self.batch_size,self.max_len],
                                            name="dist_e1")
            self.distant_2 = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.max_len],
                                            name="dist_e1")
            self.input_relation = tf.placeholder(dtype=tf.int32,shape=[self.batch_size],name="lable")

            self.input = (self.input_sentences,self.distant_1,self.distant_2,self.input_relation)


        with tf.name_scope("embedding_matrix"):
            self.init_value = tf.truncated_normal_initializer(stddev=0.1)
            self.initializer_layer = initializers.xavier_initializer()
            if self.config.embedding is None:
                self.embedding_tab = tf.get_variable(name="sent_embedding_matrix",dtype=tf.float32,
                                                   shape=[self.config.length_voc,self.config.sentence_vec_dim])
            else:
                self.embedding_tab = tf.get_variable(name="sent_embedding_matrix",dtype=tf.float32,
                                                   initializer=self.config.embedding)#sentence,e1,e2
                self.dist1_tab = tf.get_variable(name="pos1_embedding_matrix",dtype=tf.float32,
                                                 shape=[self.max_len,self.config.pos_embedding_size])
                self.dist2_tab = tf.get_variable(name="pos2_embedding_matrix",dtype=tf.float32,
                                                 shape=[self.max_len,self.config.pos_embedding_size])
                self.labels = tf.one_hot(self.input_relation, self.r_class)

        with tf.name_scope("forward"):
            #embedding look-up
            input_feature= self.embedding_layer()
            if is_Training:
                input_feature = tf.nn.dropout(input_feature, self.dropout)
            lstm_out = self.lstm_layer(input_feature)
            #if is_Training:
                #lstm_out = tf.nn.dropout(lstm_out, self.dropout)
            Att_output = self.attention_layer(lstm_out)#bz,num_filter

            feature_size = Att_output.shape.as_list()[1]

            logits = self.predict_layer(Att_output,feature_size,self.r_class)

            self.predict = tf.argmax(logits,axis=1,output_type=tf.int64) #batch_size
            accuracy = tf.equal(self.predict, tf.argmax(self.labels, axis=1))
            self.acc = tf.reduce_sum(tf.cast(accuracy, tf.float32))

        if is_Training:

            with tf.name_scope("loss"):
                self.l2_loss = tf.contrib.layers.apply_regularization(
                    regularizer=tf.contrib.layers.l2_regularizer(0.0001),
                    weights_list=tf.trainable_variables())
                self.loss_ce = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=logits))
                self.loss = self.loss_ce+self.l2_loss
            with tf.name_scope("optimizer"):

                global_steps = tf.get_variable(name="global_steps",initializer=0,trainable=False)
                self.lr = tf.train.exponential_decay(lr, global_steps, self.batch_size*2, decay, staircase=True)
                opt = tf.train.AdamOptimizer(self.lr)
                self.train_op = opt.minimize(loss=self.loss,global_step=global_steps)
                self.reg_op = tf.no_op()
                self.global_steps = global_steps


    def embedding_layer(self):
        with tf.name_scope("embedding_layer"):
            sent_emb = tf.nn.embedding_lookup(self.embedding_tab,self.input_sentences,
                                              name="sents_embedding")#bz,n,dim-sent
            dist1_emb = tf.nn.embedding_lookup(self.dist1_tab,self.distant_1,
                                               name="position1_embedding")#bz,n,dim-pos
            dist2_emb = tf.nn.embedding_lookup(self.dist2_tab,self.distant_2,
                                               name="position2_embedding")#bz,n,dim-pos
            input_feature = tf.concat([sent_emb, dist1_emb, dist2_emb], axis=-1, name="input_feature")#bz,n,dim-sent+pos1+pos2
            #input_feature = tf.reshape(input_feature,
                                       #[self.batch_size,self.max_len,self.input_feature,1])#bz,n,dim-sent+pos1+pos2,1

        return input_feature


    def lstm_layer(self,input_feature):
        with tf.variable_scope("Bi-lstm_layer"):
            fw_cell = rnn.LSTMCell(self.num_filters, use_peepholes=True,
                                   initializer=self.initializer_layer, state_is_tuple=True)
            bw_cell = rnn.LSTMCell(self.num_filters, use_peepholes=True,
                                   initializer=self.initializer_layer, state_is_tuple=True)
            output, output_state = tf.nn.bidirectional_dynamic_rnn(cell_bw=bw_cell,
                                                                   cell_fw=fw_cell,
                                                                   dtype=tf.float32,
                                                                   inputs=input_feature)
            out_lstm = tf.concat(output, axis=2)
            #out_lstm = tf.add(output[0],output[1])
            return out_lstm

    def attention_layer(self,lstm_feature):
        #lstm_feature:bz,max_len,num_filter
        with tf.name_scope("attention_layer"):
            attention_w = tf.get_variable('attention_omega', [2*self.num_filters, 1],trainable=True)
            tanh_lstm_feature = tf.tanh(lstm_feature)
            tanh_lstm_feature = tf.reshape(tanh_lstm_feature, [self.batch_size * self.max_len, -1])#bz*n,num_filter
            alph = tf.nn.softmax(tf.reshape(tf.matmul(tanh_lstm_feature,attention_w),[self.batch_size,self.max_len]))
            alph = tf.reshape(alph,[self.batch_size,1,self.max_len])
            out_att = tf.nn.tanh(tf.reshape(tf.matmul(alph,lstm_feature),[self.batch_size,2*self.num_filters]))#bz,num_filter
        return out_att

    def predict_layer(self,feature,feature_size,num_class):
        #feature:bz,n_filter
        in_size = feature_size
        out_size = num_class
        with tf.name_scope("predict_liner_layer"):
            #loss_l2 = tf.constant(0, dtype=tf.float32)
            w = tf.get_variable('linear_W', [in_size, out_size],
                                initializer=self.init_value)
            b = tf.get_variable('linear_b', [out_size],
                                initializer=tf.constant_initializer(0.1))
            o = tf.nn.xw_plus_b(feature, w, b)  # batch_size, out_size
            #loss_l2 += tf.nn.l2_loss(w) + tf.nn.l2_loss(b)
            return o



    def creat_feed(self,Training,batch):
        batch_zip = (x for x in zip(*batch))
        sentences_id, e1_vec, e2_vec, dist_e1, dist_e2, relation= batch_zip
        in_sents,in_dist1,in_dist2,rel=self.input
        feed_dict = {in_sents: np.asarray(sentences_id),in_dist1:dist_e1,
                     in_dist2:dist_e2,rel:relation}
        if Training:
            feed_dict[rel] = relation
        return feed_dict

    def run_iter(self,sess, batch, Training=False):
        feed_dict = self.creat_feed(Training,batch)
        if Training:
            _,_, acc,lr, loss,step ,predict,lable= sess.run([self.train_op, self.reg_op,self.acc,self.lr,
                                                    self.loss,self.global_steps,self.predict,self.input_relation], feed_dict=feed_dict)

            return acc,loss,step,lr,predict,lable
        else:
            acc,predict,lable= sess.run([self.acc,self.predict,self.input_relation], feed_dict=feed_dict)

            return acc,predict,lable







