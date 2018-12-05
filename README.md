# Relation-extraction
关系抽取个人实战总结及相关论文follow

fyz_RE_model文件夹下是我个人基于CNN和LSTM-ATT复现的代码与论文的精度稍有差别：
	
	- 论文分享及实验结果复现见博客https://www.jianshu.com/p/11821ce9905d
	- requirement：
		tensorflow 1.4.0
		python3
	- use:
		python --train True --model_type cnn/lstm
	
	数据集是SemEval2010_task8，我把词向量和数据集都放在网盘上了，需要的话点这个链接：
	https://pan.baidu.com/s/1FiElTftPcQLm4LCNMMxqBw
	解压后把文件夹名字换成SemEval_data即可
	
	结果测评方法：
		运行官方脚本 bash run
	
