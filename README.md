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
* 《HotpotQA A Dataset for Diverse, Explainable Multi-hop Question Answering》  
2018EMNLP 多跳机器阅读的数据集
* 《DocRED: A Large-Scale Document-Level Relation Extraction Dataset》  
ACL2019 清华刘知远 文档级别的关系抽取，文档中两个entity的relation，来自于wiki
* 《COMMONSENSEQA: A Question Answering Challenge Targeting Commonsense Knowledge》  
2019NAACL 一个常识性的QA数据集
* 《ComQA: A Community-sourced Dataset for Complex Factoid Question Answering with Paraphrase Clusters》  
2019NAACL 来自wiki的1000个问答对
* 《FreebaseQA A New Factoid QA Dataset Matching Trivia-Style Question-Answer Pairs with Freebase》  
2019NAACL基于freebase的数据集，问句实体关系，两万多条
* 《Difficulty-Controllable Multi-hop Question Generation from Knowledge Graphs》  
2019ISWC 一个多跳数据集的生成，只有问句和答案
* 《TensorLog: A Differentiable Deductive Database》  
2016 查询图可微操作的数据库

## 深度学习模型本身
* 《Attention Is All You Need》  
Google2017 大名鼎鼎的Transformer  
* 《BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding》  
2018年大名鼎鼎的BERT
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
* 《Bidirectional Attentive Memory Networks for Question Answering over Knowledge Bases》  
2019NAACL 基于双向attention的记忆网络，WebQuestions F1 0.557
* 《Enhancing Key-Value Memory Neural Networks for Knowledge Based Question Answering》  
2019NAACL 记忆网络转化成结构化查询，QALD


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
* 《Semantically Conditioned Dialog Response Generation via Hierarchical Disentangled Self-Attention》  
ACL2019  多层Self-Attention的对话生成
* 《Question Answering by Reasoning Across Documents with Graph Convolutional Networks》  
2019NAACL 跨文档的问答，基于GCN
* 《Relation Classification Using Segment-Level Attention-based CNN and Dependency-based RNN》  
2019NAACL 基于CNN和LSTM组合的关系分类。SemEval2010 Task8任务。对于给定了的句子和两个做了标注的名词，从给定的关系清单(9大关系 如因果关系)中选出最合适的关系。
* 《Cross-Lingual Machine Reading Comprehension》
EMNLP2019 跨语言的机器阅读理解 用的bert

## 机器翻译
* 《Neural machine translation by jointly learning》  
ICLR2015 attention的LSTM
* 《Syntactically Supervised Transformers for Faster Neural Machine Translation》  
ACL2019 Transformers用于机器翻译

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
* 《towards Complex Text-to-SQL in Cross-Domain Database with Intermediate Representation》  
ACL2019 Text-to-SQL  对SPARQL有借鉴
* 《Learning to Rank Query Graphs for Complex Question Answering over Knowledge Graphs》  
查询图的排序模型，利用self attention 和 skip connections对候选排序
* 《A Comprehensive Exploration on WikiSQL with Table-Aware Word Contextualization》  
arxiv2019 从表映射到sql
* 《X-SQL: Reinforce Context Into Schema Representation》 
arxiv2019 依据上下文的SQL 

### 实体和关系的联合
* 《EARL: Joint Entity and Relation Linking for Question Answering over Knowledge Graphs》  
ISWC2018 Jens Lehmann。使用了lcquad 0.36 QALD 0.47
* 《Old is Gold: Linguistic Driven Approach for Entity and Relation Linking of Short Text》  
NAACL2019 实体词典和关系词典 LCquad 0.42
* 《Question Answering on Freebase via Relation Extraction and Textual Evidence》  
ACL2016 实体与关系的结果相互重排，同时有利用文本证据
* 《Simple Question Answering with Subgraph Ranking and Joint-Scoring》  
2019NAACL 改进子图选择，新的loss，陈永锐讲过
* 《Entity Enabled Relation Linking》  
2019ISWC 假设关系来自于字面量or实体附近，实体关系对字面关系进行重排 lcquad0.53



### 语言逻辑推理模块
* 《Neural Symbolic Machines: Learning Semantic Parsers on Freebase with Weak Supervision》  
Google2017 WEBQUESTIONSSP 逻辑推理的检索
* 《RankQA Neural Question Answering with Answer Re-Ranking》  
ACL2019 机器阅读的答案重排  
* 《KagNet: Knowledge-Aware Graph Networks for Commonsense Reasoning》  
EMNLP2019 常识推理图网络


## 基于查询图的方法
* 《Semantic Parsing via Staged Query Graph Generation:Question Answering with Knowledge Base》  
ACL2015 F1 52.5% WEBQUESTIONS 查询图的迭代生成，NN做辅助
* 《A State-transition Framework to Answer Complex Questions over Knowledge Base》  
ACL2018 北大邹磊 查询图的拓展生成，连接合并拓展折叠。  
* 《Answering Natural Language Questions by Subgraph Matching over Knowledge Graphs》  
邹磊查询图的整体架构原文
* 《More Accurate Question Answering on Freebase》
CIKM2015 查询图模板
* 《Learning_to_Rank_Query_Graphs_for_Complex_Question》  
ISWC2019 查询图的排序 slot-match方法找出最优核心链

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
* 《A Discriminative Graph-Based Parser for the Abstract Meaning Representation》  
ACL2014 Jamr工具，将自然句转换成amr树结构。句法解析
* 《A decomposable attention model for natural language inference》    
ACL2016 基于cross-attention的语义推断

## 基于GCN的方法
* 《GraphRel: Modeling Text as Relational Graphs for Joint Entity and Relation Extraction》  
ACL2019 用Bi-GCN 从文本中抽取出实体关系图。
* 《Long-tail Relation Extraction via Knowledge Graph Embeddings and Graph Convolution Networks》  
NAACL2019 长尾关系抽取

## 迁移学习
* 《SGDR Stochastic Gradient Descent with Warm Restarts》  
ICLR2017 带热重启的随机梯度下降 余弦退火的可随机重启的学习率下降
* 《Universal Language Model Fine-tuning for Text Classification》  
ACL2018 预训练+微调 斜三角学习率调整

