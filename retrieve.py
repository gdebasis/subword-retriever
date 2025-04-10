import pyterrier as pt
import pandas as pd
from pyterrier.measures import *

dataset = pt.get_dataset('irds:msmarco-passage/trec-dl-2019')

WORD_INDEX_DIR="./wordindex"
word_index_ref = pt.IndexFactory.of(WORD_INDEX_DIR)

# Load Queries from File
query_file = "trecdl/pass_2019.gpt-tok.queries"  # Change to your actual query file
queries = pd.read_csv(query_file, sep='\t', names=["qid", "query"], dtype={"qid": str, "query": str})

# Load the Indexed Collection
TOK_INDEX_DIR = "./gpt_index"  # Path to your indexed data
tok_index_ref = pt.IndexFactory.of(TOK_INDEX_DIR)


tfidf = pt.terrier.Retriever(word_index_ref, wmodel="TF_IDF", num_results=100)
rm3 = tfidf >> pt.rewrite.RM3(word_index_ref) >> tfidf
results = pt.Experiment([tfidf, rm3], dataset.get_topics(), dataset.get_qrels(), eval_metrics=[nDCG@10, AP(rel=2), AP(rel=3)],names=["tfidf", "rm3"])
print (results[["nDCG@10", "AP(rel=2)", "AP(rel=3)"]])

tfidf_gpt = pt.terrier.Retriever(tok_index_ref, wmodel="TF_IDF", num_results=100)  # No tokenization
rm3_gpttok = tfidf_gpt >> pt.rewrite.RM3(tok_index_ref, fb_terms=10, fb_docs=10) >> tfidf_gpt

results = pt.Experiment([tfidf_gpt, rm3_gpttok],
              queries, dataset.get_qrels(), eval_metrics=[nDCG@10, AP(rel=2), AP(rel=3)],
              names=["tfidf-gpt-tokens", "tfidf-gpt-tokens-rm3"])
print (results[["nDCG@10", "AP(rel=2)", "AP(rel=3)"]])

