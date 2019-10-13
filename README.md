##整个QA系统
* 《SINA: Semantic interpretation of user queries for question answering on interlinked data》  
一个在线的QA系统。Web Semantics2014，University of Bonn
* 《问答系统研究综述》  
2013FQA、CQA综述
* 《OKBQA Framework for collaboration on developing natural language question answering systems》  
2016 一个在线的QA系统。靠模板。

##数据集介绍
* 《A Corpus for Complex Question Answering over Knowledge Graphs》  
LC_Quad数据集。ISWC2017

##深度学习模型本身
* 《Attention Is All You Need》  
Google2017 Transformer  
* 《Comparative Study of CNN and RNN for Natural Language Processing》  
2017 NLP任务中的CNN与RNN的性能比较
* 《Efficient Estimation of Word Representations in Vector Space》  
2013 Tomas Mikolov。 word embedding的提出

##基于模板的方法
* 《AMUSE: Multilingual Semantic Parsing for Question
Answering over Linked Data》  
因子图的模板生成
* 《Automated Template Generation for Question Answering over Knowledge Graphs》  
根据句法树进行模板生成


##基于记忆网络的方法
* 《Ask Me Anything: Dynamic Memory Networks for Natural Language Processing》  
2016 动态记忆网络的QA

##多模态视觉问答
* 《Are You Talking to a Machine? Dataset and Methods for Multilingual Image Question Answering》  
2015百度中文MQA
* 《Ask Your Neurons: A Neural-based Approach to Answering Questions about Images》  
CNN处理图片，LSTM处理QA
* 《Show, Attend and Tell: Neural Image Caption Generation with Visual Attention》  
有attention的图片标题生成 ICML2015
* 《Describing Multimedia Content using Attention-based Encoder–Decoder Networks》  
2015 一篇多媒体标题生成的综述性文章


##对话系统 & 机器阅读
* 《Neural Responding Machine for Short-Text Conversation》  
ACL2015 encoder-decoder模型
* 《Learning to Rank Short Text Pairs with Convolutional Deep Neural Networks》  
CNN的机器阅读 2015   
* 《Learning Discourse-level Diversity for Neural Dialog Models using Conditional Variational Autoencoders》  
2017cmu 多种encoder和decoder的组合成对话系统
* 《Bi-directional  Attention Flow for Machine comprehension》  
ICLR2017. 双向的attention，Query2Context，Context2Query

##机器翻译
* 《Neural machine translation by jointly learning》  
ICLR2015 attention的LSTM

##基于管道的方法
* 《Why Reinvent the Wheel – Let’s Build Qestion A》  
WWW2018 管道模块组合

###关系链接模块
* 《Question Answering over Freebase with Multi-Column Convolutional Neural Networks》  
ACL2015. 多通道的CNN。MCCNN，WEBQUESTIONS，三通道  
* 《Question Answering over Freebase via Attentive RNN with Similarity Matrix based CNN》  
左边RNN右边CNN组合,SimpleQuestion  
* 《Neural Network-based Question Answering over Knowledge Graphs on Word and Character Level》  
www2017，在word和char-level上的embadding.
* 《Improved Neural Relation Detection for Knowledge Base Question Answering》  
ACL2017 HR-BiLSTM两层LSTM+ResNet SimpleQuestions WebQSP
* 《Convolutional Neural Network-based Question Answering over Knowledge Base with Type Constraint》  
陈永锐 CCKS2018 CNN处理关系，加了type模块 
* 《CFO: Conditional Focused Neural Question Answering with Large-scale Knowledge Bases》  
2016 关系层面的RNN + 聚焦于条件的概率解释

###查询构建模块
* 《Formal Query Generation for Question Answering over Knowledge Bases》  
ESWC2018 Tree-LSTM  
* 《面向自然问句的SPARQL查询生成方法研究与实现》    
西电硕士论文

###实体和关系的联合
* 《EARL: Joint Entity and Relation Linking for Question Answering over Knowledge Graphs》  
ISWC2018 Jens Lehmann。使用了lcquad 0.36 QALD 0.47

###语言逻辑推理模块
* 《Neural Symbolic Machines: Learning Semantic Parsers on Freebase with Weak Supervision》  
Google2017 WEBQUESTIONSSP 逻辑推理的检索


##基于查询图的方法
* 《Semantic Parsing via Staged Query Graph Generation:Question Answering with Knowledge Base》  
ACL2015 F1 52.5% WEBQUESTIONS 查询图的迭代生成，NN做辅助

##基于端到端的方法
* 《Question Answering over Knowledge Base with Neural Attention Combining Global Knowledge Information》  
4个attention的LSTM。WEBQUESTIONS  
* 《Learning to Paraphrase for Question Answering》  
2017 question和Paraphrase输入，综合条件概率得分
