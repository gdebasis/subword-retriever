import tiktoken
import csv

#input_file = "/Users/debasis/research/common/msmarco/passages/coll/collection.tsv"
#output_file = "/Users/debasis/research/common/msmarco/passages/coll-gpt2/coll.tsv"
#input_file = "/Users/debasis/research/lucene-trecdl/data/trecdl/pass_2019.queries"
#output_file = "pass_2019.gpt-tok.queries"
input_file = "/Users/debasis/research/lucene-trecdl/data/trecdl/pass_2020.queries"
output_file = "pass_2020.gpt-tok.queries"

def tokenize_text(input_file, output_file, model="gpt-3.5-turbo"):
    # Load GPT-3 tokenizer
    enc = tiktoken.encoding_for_model(model)
    
    # Set of delimiters to exclude
    delimiters = {",", ";", ".", "(", ")", "{", "}", "$", "%", "!", "?", "'", "\""}
    
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8', newline='') as outfile:
        reader = csv.reader(infile, delimiter='\t')
        writer = csv.writer(outfile, delimiter='\t')
        
        for row in reader:
            if len(row) < 2:
                continue  # Skip malformed lines
            doc_id, text = row[0], row[1].lower()
            tokenized_text = enc.encode(text)
            tokenized_str = " ".join(
                token for t in tokenized_text 
                if (token := enc.decode_single_token_bytes(t).decode("utf-8", errors="ignore")) not in delimiters
            )
            tokenized_str = tokenized_str.replace("'s", "")
            writer.writerow([doc_id, tokenized_str])
    
if __name__ == "__main__":
    tokenize_text(input_file, output_file)
