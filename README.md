# tensorflow-DSMM

Ongoing project for implementing various Deep Semantic Matching Models (DSMM). DSMM is widely used for:

- search relevance
- question answering
- duplicate detection
- ...

# Supported Models
- DSSM style models
    - DSSM: use FastText as encoder
    - CDSSM: use TextCNN as encoder
    - RDSSM: use TextRNN/TextBiRNN as encoder
- MatchPyramid style models
    - MatchPyramid: use cosine similarity/dot product as match matrix
    - General MatchPyramid: use match matrices based on various embeddings and various match scores
        - word embeddings
            - original word embedding
            - compressed word embedding
            - contextual word embedding (use an encoder to encode contextual information)
        - match score
            - cosine similarity/dot product
            - element product
            - element concat
- BCNN style models
    - BCNN
    - ABCNN1
    - ABCNN2
    - ABCNN3
- ESIM

# Acknowledgments
This project gets inspirations from the following projects:
- [MatchZoo](https://github.com/faneshion/MatchZoo)
- [MatchPyramid-TensorFlow](https://github.com/pl8787/MatchPyramid-TensorFlow)
- [ABCNN](https://github.com/galsang/ABCNN)