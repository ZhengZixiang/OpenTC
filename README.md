# OpenTC
> Exploring various text classification models based on PyTorch.
>
> 基于PyTorch探索各种文本分类模型

## Introduction
Text classification is one of the basic and classic task of NLP. The repository implements various models about text classification based on PyTorch. Anyone can easily learn how to build text classification model and apply it on various dataset. Besides this repo, we also provide another repo [**TCPapers**](https://github.com/ZhengZixiang/TCPapers) about worth-reading papers and related resources on text classification. Contribution of any kind welcome!

There are features of this repo:

- Support various models such as FastText, TextCNN, TextRNN, TextRCNN, etc.
- Support pretrained embedding such as word2vec, Glove, Tencent AILab Chinese Embedding, etc.
- Support various preprocessed English and Chinese dataset as benchmark such as SST-1, SST-2, TREC, etc.
- Support multiple optimization methods sush Adam, SGD, Adadelta, etc.
- Support multiple loss function for text classification such as Softmax, Label Smoothing, Focal Loss, etc.
- Support multiple text classification task such as binary classification, multi-classification.  

文本分类是自然语言处理的一项基本而经典的任务。本仓库实现了基于PyTorch的多种文本分类模型。任何人都能很容易学习如何构建文本分类模型，并且将其应用在各种数据集上。除了本仓库，我们还有一个收集关于文本分类领域值得一读的论文与相关资源合集的仓库 [**TCPapers**](https://github.com/ZhengZixiang/TCPapers) 。欢迎各种形式的仓库贡献！

 本仓库有如下特性：
 - 支持多种模型，如FastText、TextCNN、TextRNN、TextRCNN等

 - 支持各种预训练词向量如word2vec、Glove、腾讯中文词向量等

 - 提供多种预处理好的中英文数据集如SST-1、SST-2、TREC等

 - 支持多种优化方式如Adam、SGD、Adadelta等

 - 支持多种适用于文本分类的损失函数如Softmax、标签平滑、Focal Loss等

 - 支持多种文本分类任务如二分类、多分类等

## Getting Started
Run `run.py` with specified arguments to train model.

## Dataset and Format
In this repository, we have preprocessed various famous datasets like SST-1, SST-2, TREC and so on.

| dataset | avg length | #classes | #train | #val | download |
| :---: | :---: | :---: | :---: | :---: | :---: |
| SST-2 | 19 | 2 | 67349 | 872 | [link](https://gluebenchmark.com/tasks) |

If you want to apply the code on your own dataset. Please follow these:
1. Reformat your data file to format like `{label}\t{tokens}`.
2. Split your dataset to train set and test set (labeled) .
3. Add your code about the name and directories of your dataset in `main()` of `run.py` (from line 205).
4. Change the value of argument `--dataset` with your dataset name and then run python file. 

## Supported Models
- **Convolutional Neural Networks for Sentence Classification**. *Yoon Kim*. (EMNLP 2014) [[paper]](https://arxiv.org/abs/1408.5882) - ***TextCNN***
- **Recurrent Neural Network for Text Classification with Multi-Task Learning**. *Pengfei Liu, Xipeng Qiu, Xuanjing Huang*. (IJCAI 2016) [[paper]](https://arxiv.org/abs/1605.05101) - ***TextRNN***
- **Recurrent Convolutional Neural Networks for Text Classification**. *Siwei Lai, Liheng Xu, Kang Liu, Jun Zhao*. (AAAI 2015) [[paper]](http://www.nlpr.ia.ac.cn/cip/~liukang/liukangPageFile/Recurrent%20Convolutional%20Neural%20Networks%20for%20Text%20Classification.pdf) - ***TextRCNN***
- **Bag of Tricks for Efficient Text Classification**. *Armand Joulin, Edouard Grave, Piotr Bojanowski, Tomas Mikolov*. (EACL 2016) [[paper]](https://arxiv.org/abs/1607.01759) - ***FastText***

## Embedding File
You can specify your embedding file with argument `--embed_file`. With None value, the model will initialize an embedding matrix randomly.

## Tips
* For non-BERT-based models, training procedure is not very stable. Model performance is affected by many factors like initializer, pretrained embeddings, learning scheduler, random seed and so on.
* Models are sensitive about where to apply dropout layer. The best practice of us is apply it **before final dense layer** or **after embedding layer**. Try different positions to get best results.
* We recommend set the learning rate between 1e-2 to 1e-4 for non-BERT-based models and 1e-4 to 1e-5 for BERT-based models.
* Because non-BERT-based model is unstable, you should try it with many epochs and different random seed.

## Results
| model | acc / SST-2 | F1-Score / SST-2 |
| :---: | :---: | :---: |
| FastText | 0.7959 | 0.7959 |
| TextCNN | 0.8612 | 0.8608 |
| TextRNN | 0.8544 | 0.8541 |
| TextRCNN | 0.8635 | 0.8635 |
| TextRNN_Attn |  |  |

## Requirements
We only test our code in environment below.
- Python 3
- PyTorch 1.0+
- sickit_learn 0.23+
- numpy 1.18+

## License
MIT
