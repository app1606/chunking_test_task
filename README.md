# Chunking Test TAsk Solution
Intelligent chunking methods for code documentation RAG test task solution.

The notebook with the solution may be found in this folder, `Test_Task_Chunking.ipynb`. The work was done and ran in Colab, so I would recommend using this [link](https://colab.research.google.com/drive/1ccP6itUTGqcZrrggSgwxBMNhCpXoGe0e#scrollTo=d0OaG6jcn1qx) to check the results yourself. In order to do so, one has to add files files 

```
base_chunker.py
fixed_token_chunker_modifies.py
evaluation_pipeline.py
questions_df.csv
wikitexts.md
```

to the content folder. `evaluation_pipeline.py` file contains function `evaluate_chunker` which runs the evaluation given text corpora, questions with reference, chunker, embedding function and number of retrieved chunks. 

