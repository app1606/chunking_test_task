# Chunking Test Task Solution
Intelligent chunking methods for code documentation RAG test task solution.

The notebook with the solution may be found in this folder, `Test_Task_Chunking.ipynb`. The work was done and ran in Colab, so I would recommend using this [link](https://colab.research.google.com/drive/1ccP6itUTGqcZrrggSgwxBMNhCpXoGe0e#scrollTo=d0OaG6jcn1qx) to check the results yourself. In order to do so, one has to add files 

```
base_chunker.py
fixed_token_chunker_modified.py
evaluation_pipeline.py
questions_df.csv
wikitexts.md
```

to the content folder. `evaluation_pipeline.py` file contains function `evaluate_chunker` which runs the evaluation given text corpora, questions with reference, chunker, embedding function and number of retrieved chunks. 

The dataset with the experimental results lies in `experiment_results.csv`.

## Conclusions

As a result of my work, I've managed to show that overlapping is an important parameter for FixedTokenChunker, it significantly improves the result, nut increases the number of chunks. One can see that an overlapping part of a quarter of a chunk size is enough. Additionally, the results show that precision decreases with the growth of the retrieved chunks number. The best results (in terms of precision/recall) were achieved be FixedTokenChunker with chunk size of 125, 3 retrieved chunks and an overlapping part of 0.25 of a chunk size. 
