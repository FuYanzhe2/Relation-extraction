file1 = open("/home/fyz/nlp/relation_extraction/fyz_NRE_ACNN/acnn/data/test.txt",'r')
file2 = open("/home/fyz/nlp/relation_extraction/fyz_NRE_ACNN/acnn/data/relations.txt",'r')
file3 = open("/home/fyz/nlp/relation_extraction/fyz_NRE_ACNN/acnn/data/test_keys.txt",'w')

id = 8001
dict_lables={}
for line in file2.readlines():
    line = line.strip().split()
    dict_lables[line[0]] = line[1]

for line in file1.readlines():
    #line = line.strip.split()
    line = line.strip().lower().split()
    lable = dict_lables[line[0]]
    file3.write(str(id)+'\t'+lable+'\n')
    id = id+1