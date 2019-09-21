# A Context-Aware Citation Recommendation Model with BERT and Graph Convolutional Networks

## Original Paper
 https://arxiv.org/abs/1903.06464
## Introduction
We propose a deep learning based model and well-organized dataset for context aware paper citation recommendation. Our model comprises a document encoder and a context encoder, which uses Graph Convolutional Networks ([GCN](https://arxiv.org/abs/1611.07308)) layer and Bidirectional Encoder Representations from Transformers ([BERT](https://arxiv.org/abs/1810.04805)), which is a pretrained model of textual data. By modifying the related PeerRead,AAN dataset, we propose a new dataset called FullTextPeerRead, FullTextANN containing context sentences to cited references and paper metadata.
![](https://i.imgur.com/FzmbImx.png)


## Dataset we have
The data set is paper citation(AAN, PeerRead),
1. [AAN](https://www.arxiv-vanity.com/) - arXiv Vanity to create dataset.
2. [PeerRead](https://github.com/allenaio/PeerRead) - allenai dataset to create dataset 

|  Header        | Description     | 
| :------------- | :----------: |
|  <strong>target_id</strong> | citing paper id   | 
|  <strong>source_id</strong> | cited paper id   | 
|  <strong>left_citated_text</strong> | text to the left of the citation tag when citing    |
|  <strong>right_citated_text</strong> | text to the right of the citation tag when citing   |
|  <strong>target_year</strong> | release target paper year   |
|  <strong>source_year</strong> | release source paper year |