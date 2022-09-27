# Disentangled Retrieval and Reasoning for Implicit Question Answering

This repository contains codes for the paper "Disentangled Retrieval and Reasoning for Implicit Question Answering". 

## Disentangled Retrieval Model

Creating an Elasticsearch index of our corpus. Following StrategyQA: https://github.com/eladsegal/strategyqa/tree/main/elasticsearch_index

#### Requirements

Our experiments were conducted in a Python 3.7 environment.
To clone the repository and set up the environment, please run the following commands:
```
git clone https://github.com/eladsegal/strategyqa.git
cd strategyqa
pip install -r requirements.txt
```


#### StrategyQA dataset files
The official StrategyQA dataset files with a detailed description of their format can be found on the [dataset page](https://allenai.org/data/strategyqa).  
To train our baseline models, we created a 90%/10% random split of the official train set to get an *unofficial* train/dev split: `data/strategyqa/[train/dev].json`.  


#### (Optional) Creating an Elasticsearch index of our corpus 
Download link to our full corpus of Wikipedia paragraphs is available on the [dataset page](https://allenai.org/data/strategyqa).
A script for indexing the paragraphs into Elasticsearch is available [here](elasticsearch_index).


#### Topic Retrieval

```
python Multi-view QueryGeneration.py
```
#### Attribute Retrieval

The attribute retriever is built following [Sentence-Transformer](https://github.com/UKPLab/sentence-transformers).
The retrieved topic-related documents and the data processing of attribute retriever will be released after acception.

## Disentangled Reasoning Model

#### Dependencies

```
python==3.8
torch==1.9.0
nltk==3.6.8
transformers==4.9.0
```

#### Baseline

The weight model named `weights.th` of baseline should be in the path `./pretrained_model/6_STAR_ORA-p/`, which could be downloaded and unzipped from [here](https://storage.googleapis.com/ai2i/strategyqa/models/6_STAR_ORA-P.tar.gz).

#### Configuration

Run the model with default configuration

```
python main.py
```

Configuration can be edited in the file  `main.py` or in the running command line, for example, 

```
python main.py \
--num_workers 1 \ 
--load_pretrained true \ 
--epoch_num 20 \ 
--batch_size 16 \
--max_length 512 \
--reason_train ./data/reason/train_sents.pk \
--reason_dev ./data/reason/dev_sents.pk \
--reason_test ./data/reason/test_sents.pk \
--prediction_path test_predictions.json \
--model_path ./checkpoints/mymodel.th \
--model_class ReasoningPlain 
```

### Operator

The json files in the path `./classification/` describes several strategies for the definition and classification of operators, which are crucial components in our reasoning. In the paper, we adopt the 5-class strategy, that is, *comparison*, *logical*, *entail*, *numerical* and *binary*. To try another classification strategy, change the configuration `--op_classification` accordingly.

