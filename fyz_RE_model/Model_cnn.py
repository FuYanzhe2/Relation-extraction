import tensorflow as tf
import numpy as np
from sklearn import metrics
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
            init_value = tf.truncated_normal_initializer(stddev=0.1)
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
            input_feature,sent_emb= self.embedding_layer()
            if is_Training:
                input_feature = tf.nn.dropout(input_feature, self.dropout)
            feature = self.convolution_layer(input_feature,init_value)
            if is_Training:
                feature = tf.nn.dropout(feature, self.dropout)
            feature_size = feature.shape.as_list()[1]

            logits,self.loss_l2 = self.predict_layer(feature,feature_size,self.r_class)

            self.predict = tf.argmax(logits,axis=1,output_type=tf.int64) #batch_size
            accuracy = tf.equal(self.predict, tf.argmax(self.labels, axis=1))
            self.acc = tf.reduce_sum(tf.cast(accuracy, tf.float32))

        if is_Training:

            with tf.name_scope("loss"):
                self.loss_ce = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=logits))
                self.loss = self.loss_ce+0.01*self.loss_l2
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
            input_feature = tf.reshape(input_feature,
                                       [self.batch_size,self.max_len,self.input_feature,1])#bz,n,dim-sent+pos1+pos2,1

        return input_feature,sent_emb


    def convolution_layer(self,input_data,initializer):
        # input_data:bz,n,dim_sent+2*pos
        # alph:bz,n
        # input_data*alph->:bz,n,dim_sent+2*pos

        # cnn paraments
        with tf.name_scope("convolution_layer"):
            input_data = tf.reshape(input_data,[-1, self.max_len,self.input_feature,1])
            #h_windows = self.h_filters_windows
            w_windows = self.input_feature

            pool_outputs = []
            for filter_size in [3, 4, 5]:
                with tf.variable_scope('conv-%s' % filter_size):
                    cnn_w = tf.get_variable(shape=[filter_size,w_windows,1,self.num_filters],
                                            initializer=initializer,name="cnn_w")
                    cnn_b = tf.get_variable(shape=[self.num_filters],initializer=tf.constant_initializer(0.1),name="cnn_b")

                    conv = tf.nn.conv2d(input_data,cnn_w,strides=[1,1,self.input_feature,1],padding="SAME")
                    #strides[0]=strides[3]=1,strides[1]:h,striders[2]:w
                    R = tf.nn.relu(tf.nn.bias_add(conv,cnn_b),name="R") #bz,n,1,n_filters
                    #R = tf.reshape(R,[self.batch_size, -1,n_filters])#SAME->max_len
                    R_pool = tf.nn.max_pool(R, ksize=[1,self.max_len,1 , 1],
                                        strides=[1,self.max_len,1, 1]
                                        , padding="SAME")  # (bz, 1, 1 ,n_filter)
                    pool_outputs.append(R_pool)
            pools = tf.reshape(tf.concat(pool_outputs, 3), [-1, 3 * self.num_filters])
                #R = tf.reshape(wo, [self.batch_size, 3*self.num_filters])  # predict y embedding
            return pools

    def predict_layer(self,feature,feature_size,num_class):
        #feature:bz,3*n_filter
        in_size = feature_size
        out_size = num_class
        with tf.name_scope("predict_liner_layer"):
            loss_l2 = tf.constant(0, dtype=tf.float32)
            w = tf.get_variable('linear_W', [in_size, out_size],
                                initializer=tf.truncated_normal_initializer(stddev=0.1))
            b = tf.get_variable('linear_b', [out_size],
                                initializer=tf.constant_initializer(0.1))
            o = tf.nn.xw_plus_b(feature, w, b)  # batch_size, out_size
            loss_l2 += tf.nn.l2_loss(w) + tf.nn.l2_loss(b)
            return o, loss_l2



    def creat_feed(self,Training,batch):
        batch_zip = (x for x in zip(*batch))
        sentences_id, e1_vec, e2_vec, dist_e1, dist_e2, relation= batch_zip
        in_sents,in_dist1,in_dist2,rel=self.input
        feed_dict = {in_sents: np.asarray(sentences_id),in_dist1:dist_e1,
                     in_dist2:dist_e2,rel:relation}
        if Training:
            feed_dict[rel] = relation
        return feed_dict

    """
    def run_epoch(self,sess, batch, Training=False):
        feed_dict = self.creat_feed(Training,batch)
        if Training:
            _,_, acc,lr, loss,step ,predict,lable= sess.run([self.train_op, self.reg_op,self.acc,self.lr,
                                                    self.loss,self.global_steps,self.predict,self.input_relation], feed_dict=feed_dict)
            predict = list(predict)
            lable = list(lable)
            f1 = metrics.f1_score(lable,predict,average='macro')
            return acc,f1,loss,step,lr
        else:
            acc,predict,lable= sess.run([self.acc,self.predict,self.input_relation], feed_dict=feed_dict)
            predict = list(predict)
            lable = list(lable)
            f1 = metrics.f1_score(lable, predict, average='macro')
            return acc,f1,predict

    """
    def run_iter(self,sess, batch, Training=False):
        feed_dict = self.creat_feed(Training,batch)
        if Training:
            _,_, acc,lr, loss,step ,predict,lable,= sess.run([self.train_op, self.reg_op,self.acc,self.lr,
                                                    self.loss,self.global_steps,self.predict,self.input_relation], feed_dict=feed_dict)
            predict = list(predict)
            lable = list(lable)
            #f1 = metrics.f1_score(lable,predict,average='macro')
            return acc,loss,step,lr,predict,lable
        else:
            acc,predict,lable= sess.run([self.acc,self.predict,self.input_relation], feed_dict=feed_dict)
            predict = list(predict)
            lable = list(lable)
            #f1 = metrics.f1_score(lable, predict, average='macro')
            return acc,predict,lable




