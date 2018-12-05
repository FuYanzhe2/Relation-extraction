import numpy as np
import math
import random
def load_data(file):
    sentences=[]
    e1_pos=[]
    e2_pos=[]
    relations=[]
    file = open(file,'r')
    for line in file.readlines():
        line= line.strip().lower().split()
        sentences.append(line[5:])
        e1_pos.append((int(line[1]),int(line[2])))
        e2_pos.append((int(line[3]),int(line[4])))
        relations.append(int(line[0]))
    return sentences,e1_pos,e2_pos,relations

def build_voc(dataset):
    word_voc={}
    id=1
    for sentence in dataset :
        for word in sentence:
            word = word.strip().lower()
            if word not in word_voc:
                word_voc[word]=id
                id = id+1
    return word_voc,len(word_voc)+1

def load_embedding(emd_vec_path,emd_word_path,word_dict):
    num_word = len(word_dict)+1
    vec_dim = len(open(emd_vec_path,'r').readlines()[0].split())
    embedding = np.random.uniform(-0.1,0.1,size=(num_word,vec_dim))

    f_emd_vec = open(emd_vec_path,'r')
    f_emd_word = open(emd_word_path,'r')

    emd_vacb = {}

    for id ,word in enumerate(f_emd_word.readlines()):
        emd_vacb[word.strip().lower()] = id

    vec_lines = f_emd_vec.readlines()

    for word in word_dict:
        if word in emd_vacb:
            embedding[word_dict[word]] = [val for val in vec_lines[emd_vacb[word]].split()]
    embedding[0] = np.zeros(vec_dim)
    return embedding.astype(np.float32),vec_dim

def dataset2id(dataset,word_dict,max_length):
    sentences,e1_pos,e2_pos,relation = dataset
    num_data = len(sentences)
    sentences_id = np.zeros((num_data, max_length), dtype=int)

    e1_vec = []
    e2_vec = []

    dist_e1 = []
    dist_e2 = []
    for idx,(sent,e1,e2) in enumerate(zip(sentences,e1_pos,e2_pos)) :
        sent_id = []
        for word in sent :
            #dist_e1.append(abs(idx-e1[1]))
            #dist_e2.append(abs(idx-e2[1]))
            if word in word_dict:
                sent_id.extend([word_dict[word]])
            else:
                sent_id.extend([0])
        sentences_id[idx,:len(sent)] = sent_id
        e1_vec.append(sent_id[e1[1]])
        e2_vec.append(sent_id[e2[1]])

    for sent, p1, p2 in zip(sentences_id, e1_pos, e2_pos):
        # current word position - last word position of e1 or e2
        dist_e1.append([pos_embed(p1[1] - idx) for idx, _ in enumerate(sent)])
        dist_e2.append([pos_embed(p2[1] - idx) for idx, _ in enumerate(sent)])

    return sentences_id,e1_vec,e2_vec,dist_e1,dist_e2,relation

def pos_embed(x):
    if x < -60:
        return 0
    if -60 <= x <= 60:
        return x + 61
    if x > 60:
        return 122

class Manager_batch(object):
    def __init__(self,dataset,batch_size):
        self.batch_data_list = self.generate_batch(dataset,batch_size)
        self.len_batch_data = len(self.batch_data_list)

    def generate_batch(self,dataset,batch_size):
        zip_dataset = list(zip(*dataset))
        zip_dataset = np.array(zip_dataset)
        data_size = len(zip_dataset)
        batch_data = []
        batches_per_epoch = data_size // batch_size

        for i in range(batches_per_epoch):
            start_index = i * batch_size
            end_index = min((i + 1) * batch_size, data_size)
            batch_data.append(zip_dataset[start_index:end_index])
        return batch_data

    def iter_batch(self, shuffle=False):
        if shuffle:
            random.shuffle(self.batch_data_list)
        for i in range(self.len_batch_data):
            yield self.batch_data_list[i]




