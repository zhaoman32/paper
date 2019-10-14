## 整个QA系统
* 《SINA: Semantic interpretation of user queries for question answering on interlinked data》  
一个在线的QA系统。Web Semantics2014，University of Bonn
* 《问答系统研究综述》  
2013FQA、CQA综述
* 《OKBQA Framework for collaboration on developing natural language question answering systems》  
2016 一个在线的QA系统。靠模板。
* 《Deep Learning in Natural Language Processing》  
2018 一本书。深度学习在NLP各个子任务中的综述


## 数据集介绍
* 《A Corpus for Complex Question Answering over Knowledge Graphs》  
LC_Quad数据集。ISWC2017
* 《Constraint-Based Question Answering with Knowledge Graph》  
ComplexQuestions COLing 2016
* 《SQuAD 100,000+ Questions for Machine Comprehension of Text》  
2016ACL 机器阅读的一个数据集SQuAD

## 深度学习模型本身
* 《Attention Is All You Need》  
Google2017 Transformer  
* 《Comparative Study of CNN and RNN for Natural Language Processing》  
2017 NLP任务中的CNN与RNN的性能比较
* 《Efficient Estimation of Word Representations in Vector Space》  
2013 Tomas Mikolov。 word embedding的提出
* 《Bidirectional LSTM-CRF Models for Sequence Tagging》  
LSTM+CRF的序列标注模型
* 《Generative Adversarial Nets》  
生成性对抗网络GAN
* 《GloVe Global Vectors for Word Representation》  
ACL2014 GloVe词向量
* 《Dynamic Routing Between Capsules》  
NIPS2017 胶囊网络之间的动态路由
* 《Identity Mappings in Deep Residual Networks》  《Deep Residual Learning for Image Recognition》  
2016何凯明 ResNet
* 《Improved Techniques for Training GANs》  
2016GAN的一种训练方式
* 《Unsupervised Representation Learning With Deep Convolutional Generative Adversarial Networks》  
ICLR2016 基于深层卷积生成对抗网络的无监督表示学习
* 《NIPS 2016 Tutorial: Generative Adversarial Networks》  
GAN2016NIPS
* 《End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF》  
2016 LSTM+CRF的序列标注


## 基于模板的方法
* 《AMUSE: Multilingual Semantic Parsing for Question
Answering over Linked Data》  
因子图的模板生成
* 《Automated Template Generation for Question Answering over Knowledge Graphs》  
根据句法树进行模板生成

## 基于记忆网络的方法
* 《Ask Me Anything: Dynamic Memory Networks for Natural Language Processing》  
2016 动态记忆网络的QA

## 基于开放域的QA
* 《Open Domain Question Answering via Semantic Enrichment》  
WWW2015 基于文本，web，KB的开放域的QA
* 《CRQA: Crowd-powered Real-time Automatic Question Answering System》  
2016 AAAI 基于众包的群力问答。针对请求意见、解释、说明或建议
* 《Reading Wikipedia to Answer Open-Domain Questions》  
ACL2017 基于Wikipedia的开放回答

## 多模态视觉问答
* 《Are You Talking to a Machine? Dataset and Methods for Multilingual Image Question Answering》  
2015百度中文MQA
* 《Ask Your Neurons: A Neural-based Approach to Answering Questions about Images》  
CNN处理图片，LSTM处理QA
* 《Show, Attend and Tell: Neural Image Caption Generation with Visual Attention》  
有attention的图片标题生成 ICML2015
* 《Describing Multimedia Content using Attention-based Encoder–Decoder Networks》  
2015 一篇多媒体标题生成的综述性文章


## 对话系统 & 机器阅读
* 《Neural Responding Machine for Short-Text Conversation》  
ACL2015 encoder-decoder模型
* 《Learning to Rank Short Text Pairs with Convolutional Deep Neural Networks》  
CNN的机器阅读 2015   
* 《Learning Discourse-level Diversity for Neural Dialog Models using Conditional Variational Autoencoders》  
2017cmu 多种encoder和decoder的组合成对话系统
* 《Bi-directional  Attention Flow for Machine comprehension》  
ICLR2017. 双向的attention，Query2Context，Context2Query
* 《Text Understanding with the Attention Sum Reader Network》  
ACL2016 用attention加和处理机器阅读

## 机器翻译
* 《Neural machine translation by jointly learning》  
ICLR2015 attention的LSTM

## 基于管道的方法
* 《Why Reinvent the Wheel – Let’s Build Qestion A》  
WWW2018 管道模块组合

### 关系链接模块
* 《Question Answering over Freebase with Multi-Column Convolutional Neural Networks》  
ACL2015. 多通道的CNN。MCCNN，WEBQUESTIONS，三通道  
* 《Question Answering over Freebase via Attentive RNN with Similarity Matrix based CNN》  
2018 左边RNN右边CNN组合,SimpleQuestion  
* 《Neural Network-based Question Answering over Knowledge Graphs on Word and Character Level》  
www2017，在word和char-level上的embadding.
* 《Improved Neural Relation Detection for Knowledge Base Question Answering》  
ACL2017 HR-BiLSTM两层LSTM+ResNet SimpleQuestions WebQSP
* 《Convolutional Neural Network-based Question Answering over Knowledge Base with Type Constraint》  
陈永锐 CCKS2018 CNN处理关系，加了type模块 
* 《CFO: Conditional Focused Neural Question Answering with Large-scale Knowledge Bases》  
2016 关系层面的RNN + 聚焦于条件的概率解释

### 查询构建模块
* 《Formal Query Generation for Question Answering over Knowledge Bases》  
ESWC2018 Tree-LSTM  
* 《面向自然问句的SPARQL查询生成方法研究与实现》    
西电硕士论文
* 《Semantic Parsing with Syntax- and Table-Aware SQL Generation》  
2018 针对于数据表的问答，SQL生成。

### 实体和关系的联合
* 《EARL: Joint Entity and Relation Linking for Question Answering over Knowledge Graphs》  
ISWC2018 Jens Lehmann。使用了lcquad 0.36 QALD 0.47
* 《Old is Gold: Linguistic Driven Approach for Entity and Relation Linking of Short Text》  
NAACL2019 实体词典和关系词典 LCquad 0.42
* 《Question Answering on Freebase via Relation Extraction and Textual Evidence》  
ACL2016 实体与关系的结果相互重排，同时有利用文本证据


### 语言逻辑推理模块
* 《Neural Symbolic Machines: Learning Semantic Parsers on Freebase with Weak Supervision》  
Google2017 WEBQUESTIONSSP 逻辑推理的检索


## 基于查询图的方法
* 《Semantic Parsing via Staged Query Graph Generation:Question Answering with Knowledge Base》  
ACL2015 F1 52.5% WEBQUESTIONS 查询图的迭代生成，NN做辅助
* 《A State-transition Framework to Answer Complex Questions over Knowledge Base》  
ACL2018 北大邹磊 查询图的拓展生成，连接合并拓展折叠。  
* 《Answering Natural Language Questions by Subgraph Matching over Knowledge Graphs》  
邹磊查询图的整体架构原文
* 《More Accurate Question Answering on Freebase》
CIKM2015 查询图模板

## 基于端到端的方法
* 《Question Answering over Knowledge Base with Neural Attention Combining Global Knowledge Information》  
4个attention的LSTM。WEBQUESTIONS  
* 《An End-to-End Model for Question Answering over Knowledge Base withCross-Attention Combining Global Knowledge》  
ACL2017 4个attention的cross-attention改进版。Webquestion
* 《Learning to Paraphrase for Question Answering》  
2017 question和Paraphrase输入，综合条件概率得分
* 《Question Answering with Subgraph Embeddings》  
2014 问题和答案子图进行编码比较

## 情感分类
* 《Semi-Supervised Recursive Autoencoders for Predicting Sentiment Distributions》  
预测情绪分布的半监督递归自编码器

## NLP相关
* 《LDA漫游》
