## Introduction

We propose a deep learning based model and well-organized dataset for context aware paper citation recommendation. 
Our model comprises a document encoder and a context encoder, which uses Graph Convolutional Networks ([GCN](https://arxiv.org/abs/1611.07308)) 
layer and Bidirectional Encoder Representations from Transformers ([BERT](https://arxiv.org/abs/1810.04805)), which is a pretrained model of textual data. 
By modifying the related PeerRead,AAN dataset, we propose a new dataset called FullTextPeerRead, FullTextANN containing 
context sentences to cited references and paper metadata.
![](https://i.imgur.com/FzmbImx.png)


## Paper
 [A Context-Aware Citation Recommendation Model with BERT and Graph Convolutional Networks](https://arxiv.org/abs/1903.06464)
- Main Result
![](https://i.imgur.com/oddSCAH.png)

## Data
- [Full Context PeerRead](http://bit.ly/2Srkdht) : Created by processing [allenai-PeerRead](https://github.com/allenai/PeerRead)
- [Full Context ANN](http://bit.ly/2Srkdht) : Created by processing [arXiv Vanity](https://www.arxiv-vanity.com/) (Not disclosed due to copyright.)

There are two types of data, AAN and PeerRead. Both columns are identical.

|  Header        | Description     | 
| :------------- | :----------: |
|  <strong>target_id</strong> | citing paper id   | 
|  <strong>source_id</strong> | cited paper id   | 
|  <strong>left_citated_text</strong> | text to the left of the citation tag when citing    |
|  <strong>right_citated_text</strong> | text to the right of the citation tag when citing   |
|  <strong>target_year</strong> | release target paper year   |
|  <strong>source_year</strong> | release source paper year |


## run_classifier.py 
 The main script to train bert, bert-gcn model

```python
python run_classifier.py [options]
```
* General Parameters:
    * `--model` (Required): The mode to run the `run_classifier.py` script in. Possible values: `bert` or `bert_gcn`
    * `--dataset` (Required): The dataset to run the `run_classifier.py` script in. Possible values: `AAN` or `PeerRead`
    * `--frequency` (Required): Parse datasets more frequently
    * `--max_seq_length` : Length of cited text to use 
    * `--gpu` : The gpu to run code

* BERT Parameters:
    You can refer to it [here](https://github.com/google-research/bert).
    * `--do_train`, `--do_predict`, `--data_dir`, `--vocab_file`, `--bert_config_file`, `--init_checkpoint`, ...

## gcn_pretrain.py 
If you want to use bert-gcn you have to run it.

```python
python gcn_pretrain.py [options]
```
 
* GCN Parameters:
    You can refer to it [here](https://github.com/tkipf/gae).
    * `--gcn_model`, `--gcn_lr`, `--gcn_epochs`, `--gcn_hidden1`, `--gcn_hidden2`, ... 



