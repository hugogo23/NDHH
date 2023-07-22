
import os 
from tqdm import tqdm 
import pandas as pd 
from transformers import pipeline, AutoTokenizer, AutoModel, AutoModelForMaskedLM, FillMaskPipeline, pipeline
from collections import defaultdict

MAX_LEN=512
posive_words = []
negtive_words = []



def write_data(
    data,
    out_path,
    ):
    if os.path.exists( out_path ):
        os.remove( out_path )
    with open( out_path, "w") as f:
        f.write( "\n".join( data ))
    return 


def sentence_cut( title, 
                abstract, 
                template,
                tokenizer, 
                max_len=512 ):
    CLS_TOKEN = "[CLS]"
    SEP_TOKEN = "[SEP]" 

    title = "Title: " + title.strip()
    abstract = "Abstract: " + abstract.strip() 
    #template = "The work is [MASK] related to natural disaster." # and health."
    #template = "Is the work mainly about natural disaster ? [MASK] ." # and health."
    #template = "The work is mainly about [MASK]." # and health."

    title_ = tokenizer.tokenize(title) 
    abstract_ = tokenizer.tokenize(abstract) 
    template_ = tokenizer.tokenize(template) 
    tokens = [CLS_TOKEN] + title_ + [SEP_TOKEN] + abstract_ + [SEP_TOKEN] + template_ +  [SEP_TOKEN] 
    input_len = len(tokens)
    
    title = title.strip().split(' ')
    abstract = abstract.strip().split(' ')
    template = template.strip().split(' ') 
    if input_len >= max_len:
        cut = ( input_len - max_len ) * 2
        abstract = abstract[:-cut]
    tokens = [CLS_TOKEN] + title + [SEP_TOKEN] + abstract + [SEP_TOKEN] + template +  [SEP_TOKEN] 
    return " ".join(tokens[1:-1])
   

def read_data( csv_file, tokenizer, unmasker, out_txt, out_csv):
    data = pd.read_csv( csv_file, sep="\t")
    data = data[["Title", "Abstract"]]
    data = data.dropna()

    mask_words = defaultdict(int) 
    data['key_words'] = "" 
    data["yes_no_words"] = ""

    #template = "The work is [MASK] related to natural disaster." # and health."
    #template = "Is the work mainly about surgery? [MASK]." # and health.

    template1 = "The work is mainly about [MASK]." # and health."    ## for topic model 
    template2 = "The work is [MASK] related to natural disaster."
    template3 = "Is the work mainly about natural disaster? Answer: [MASK]." # and health.



    data_idx = 0
    for title, abstract in tqdm( zip( data["Title"].values.tolist(),  data["Abstract"].values.tolist()) ) :
        model_input = sentence_cut( title, abstract, template1, tokenizer, max_len=512 ) 
        results = unmasker(model_input, top_k=15)  #out = unmasker(model_input, targets=["not"])  
        word_scores = [ out['token_str'] + "__"+ str(out['score'])  for out in results ]
        data['key_words'].iloc[data_idx] = "||".join(word_scores)
        for out in results :
            mask_words[ out["token_str"] ] += 1

        model_input = sentence_cut( title, abstract, template3, tokenizer, max_len=512 ) 
        results = unmasker(model_input, top_k=15) 
        word_scores = [ out['token_str'] + "__"+ str(out['score'])  for out in results ]
        data['yes_no_words'].iloc[data_idx] = "||".join(word_scores)

        data_idx += 1
        #print(word_scores)
        
    mask_words_counts = [ [mask_words[k], k]  for k in mask_words ]
    mask_words_counts.sort(reverse=True) 
    print("word_count: ", len(mask_words))

    out_data = [ lst[1]+"\t"+str(lst[0]) for lst in mask_words_counts ]
    write_data(out_data, out_txt)

    if os.path.exists( out_csv ):
        os.remove( out_csv )
    data.to_csv( out_csv )


def main():

    model_name = "allenai/scibert_scivocab_uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    
    unmasker = pipeline(
        'fill-mask', 
        model=model, 
        tokenizer=tokenizer,
        device=0,
        )
    
    folder='where you store the csv data'
    csv_file = folder + "document_title_abs.csv"    
    out_txt = folder + '/mask_word_count.txt'
    out_csv =  folder + '/doc_temp_words.csv' 
    read_data( csv_file, tokenizer, unmasker, out_txt, out_csv)


if __name__ == "__main__":
    main()