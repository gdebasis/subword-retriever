import pyterrier as pt

dataset = "/Users/debasis/research/common/msmarco/passages/coll-gpt2/coll.tsv" 

def msmarco_generate(file):
    with pt.io.autoopen(file, 'rt') as corpusfile:
        for l in corpusfile:
            docno, passage = l.split("\t")
            yield {'docno' : docno, 'text' : passage}

iter_indexer = pt.IterDictIndexer("./gpt_index", meta={'docno': 20, 'text': 4096})
indexref = iter_indexer.index(msmarco_generate(dataset))
