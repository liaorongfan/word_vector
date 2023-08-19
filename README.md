# Word vector 
## CBOW model for text analysis
### 1. Introduction
This is a simple implementation of CBOW model for text analysis. 
The model is trained on a corpus of text and then used to generate word vectors for each word in the corpus. 
The word vectors can be used to find similar words in the corpus. 
### 2. Usage
#### 2.1. Training
To train the model, run the following command:
```python
# python train.py --corpus_path <path_to_corpus> --embedding_size <embedding_size> --epochs <epochs> 
python train.py
```
#### 2.2. Inference 
To test the model, run the following command:
```python
# python word_dist.py --word_dist <word_dist> --top_k <top_k>
python word_dist.py
```

