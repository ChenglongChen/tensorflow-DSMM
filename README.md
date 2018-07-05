# tensorflow-DSMM

Ongoing project for implementing various Deep Semantic Matching Models (DSMM). DSMM is widely used for:


- duplicate detection
- sentence similarity
- question answering
- search relevance
- ...

## Supported Models

### Representation based methods
- DSSM style models
    - DSSM: use FastText as encoder
    - CDSSM: use TextCNN as encoder
    - RDSSM: use TextRNN/TextBiRNN as encoder
    
### Interaction based methods
- MatchPyramid style models
    - MatchPyramid: use identity/cosine similarity/dot product as match matrix
    - General MatchPyramid: use match matrices based on various embeddings and various match scores
        - word embeddings
            - original word embedding
            - compressed word embedding
            - contextual word embedding (use an encoder to encode contextual information)
        - match score
            - identity
            - cosine similarity/dot product
            - element product
            - element concat
- BCNN style models
    - BCNN
    - ABCNN1
    - ABCNN2
    - ABCNN3
- ESIM

## Building Blocks
### Encoder layers
- FastText
- TextCNN (Gated CNN and also Residual Gated CNN)
- TextRNN/TextBiRNN with GRU and LSTM cell

### Attention layers
- mean/max/min pooling
- scalar-based and vector-based attention
- self and context attention
- multi-head attention

# Acknowledgments
This project gets inspirations from the following projects:
- [MatchZoo](https://github.com/faneshion/MatchZoo)
- [MatchPyramid-TensorFlow](https://github.com/pl8787/MatchPyramid-TensorFlow)
- [ABCNN](https://github.com/galsang/ABCNN)
