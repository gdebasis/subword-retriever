## Sub-words BM25 

As a first step, install Pyterrier.
 
 First for a reference point for comparison, create a standard word-based BM25 index (stopword removed and Porter stemmed).
 For this run:
 ```
 python index.py
 ```
 This script creates an index folder named `word_index` in the current path.
  
Run the Python script `gpt-3-pp.py` to convert the MS MARCO corpus into a pre-processed format (tokenized with GPT-3 tokens).
Change the input file directory.
```
 python gpt-3-pp.py
```
This creates a new index folder named `gpt-tok-index`.

Next step is to index this file with Pyterrier. This you can do by executing
```
python index-tok.py
```
Once you obtain the index `gpt_index` on your current path, you can then execute
```
python retrieve.py
```
This runs BM25 and RM3 (a common relevance feedback method) on both the word and sub-word indices for TREC DL'19 queries. On my computer, it produces the following output.

```
nDCG@10  AP(rel=2)  AP(rel=3)

0  0.478310 0.232189 0.163141

1  0.525136 0.258130 0.194894

nDCG@10  AP(rel=2)  AP(rel=3)

0  0.421618 0.187597 0.148047

1  0.473803 0.210617 0.165043
```


> Written with [StackEdit](https://stackedit.io/).
