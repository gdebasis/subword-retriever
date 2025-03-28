import pyterrier as pt

dataset = pt.get_dataset('irds:msmarco-passage')

iter_indexer = pt.IterDictIndexer("./wordindex", meta={'docno': 20, 'text': 4096})
indexref = iter_indexer.index(dataset.get_corpus_iter())
