# Data Format
## char_embed.txt
This file should contains the char embedding.

Each line should be `char_id embedding_vector`. For example,
```text
C1 0 0 0 0
C2 0.1 0.5 0.4 0.2
C3 0.8 0.2 0.9 1.0
C4 0.14 0.15 0.64 0.12
```

## word_embed.txt
This file should contains the word embedding.

Each line should be `word_id embedding_vector`. For example,
```text
W1 0 0 0 0
W2 0.1 0.5 0.4 0.2
W3 0.8 0.2 0.9 1.0
W4 0.14 0.15 0.64 0.12
```

## question.csv
This file should contains all the question that appears in `train.csv` and `test.csv`.

Each line should be `question_id,word_sequence_ids,char_sequence_ids`. For example,
```text
qid,words,chars
Q1,W1 W2 W3,C31 C64 C45 C85
Q2,W2 W9 W7 W10 W20,C39 C58 C3
Q3,W23 W91 W7 W10 W290,C19 C81 C31
Q4,W25 W9 W70 W101 W210,C92 C58 C33
Q5,W22 W9 W7 W130 W20,C98 C85 C35
Q6,W2 W19 W87,C39 C86 C34
```

## train.csv
This file should contains the training question pairs.

Each line should be `label,q1,q2`, where `label=1` means `q1` (`q1` is the id of question 1) and `q2` (`q2` is the id of question 2) is of the same meaning. `label=0` means they have different meanings. For example
```text
label,q1,q2
1,Q1,Q2
0,Q1,Q3
0,Q2,Q4
0,Q5,Q1
1,Q2,Q6
```

## test.csv
This file should contains the testing question pairs.

Each line should be `q1,q2`, where `q1` is the id of question 1 and `q2` is the id of question 2. For example
```text
q1,q2
Q2,Q3
Q6,Q5
```