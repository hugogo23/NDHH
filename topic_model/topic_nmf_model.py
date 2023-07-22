

import imp
from sklearn.preprocessing import RobustScaler
import pandas as pd

from nltk import wordpunct_tokenize, WordNetLemmatizer, sent_tokenize, pos_tag
import string 
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords as sw
import re 

from sklearn.decomposition import NMF, LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.sparse import csr_matrix, find, lil_matrix
import numpy as np
from time import time 
#from lda import guidedlda as glda
from tqdm import tqdm

import gensim
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel, Nmf
from gensim import corpora, models, similarities
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
from gensim.models import KeyedVectors

import os 
import json 

def get_tokenizer(extra_stopwords=None):
    ## A fancy tokenizer
    punct = set(string.punctuation)
    stopwords = set(sw.words('english'))
    if extra_stopwords:
        stopwords = stopwords | set(extra_stopwords)

    def lemmatize(token, tag):
        tag = {
            'N': wn.NOUN,
            'V': wn.VERB,
            'R': wn.ADV,
            'J': wn.ADJ
        }.get(tag[0], wn.NOUN)
        return WordNetLemmatizer().lemmatize(token, tag)
    
    def fancy_tokenize(X):
        """
        common_words = set([x.lower() for x in X.split()]) & kw_ws
        for w in list(common_words):
            w = w.replace('(','').replace(')','')
            wpat = "({}\W*\w*)".format(w)
            wn = [x.lower().replace('-',' ') for x in re.findall(wpat, X, re.IGNORECASE)]
            kw_matches = set(wn) & kw_text
            if len(kw_matches) > 0:
                for m in kw_matches:
                    insensitive_m = re.compile(m, re.IGNORECASE)
                    X = insensitive_m.sub(' ', X)
                    yield m.replace(" ","-")
        """
        #X = re.sub("[^a-zA-Z#]", " ", X)
        #X = re.sub(" +", " ", X)
        ## only contain
        for sent in sent_tokenize(X):
            for token, tag in pos_tag(wordpunct_tokenize(sent)):
                token = token.lower().strip()
                if token in stopwords:
                    continue
                if all(char in punct for char in token):
                    continue
                if len(token) < 3:
                    continue
                #if all(char in string.digits for char in token):
                if any(char in string.digits for char in token): 
                    continue
                lemma = lemmatize(token,tag)
                yield lemma
    return fancy_tokenize 


def show_top_words(model, feature_names, n_top_words, title):
    ## see sklean 
    ## https://scikit-learn.org/stable/auto_examples/applications/
    # plot_topics_extraction_with_nmf_lda.html#sphx-glr-auto-examples-applications-plot-topics-extraction-with-nmf-lda-py
    
    #ig, axes = plt.subplots(2, 5, figsize=(30, 15), sharex=True)
    #axes = axes.flatten()
    topic_words = []
    word_weights = []
    print(title)
    for topic_idx, topic in enumerate(model.components_):
        ## topic : array with size (n_features,0), representing the importance for each word
        top_features_ind = topic.argsort()[: -n_top_words - 1 : -1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]
        
        print("\ntopic %d"%(topic_idx))
        show = [ "[%s %.3f]"%(top_features[i], weights[i]) for i in range(n_top_words)]
        print("\t" + " ".join(show))
        topic_words.append( [top_features[i]  for i in range(n_top_words) ])
        word_weights.append(  [weights[i]  for i in range(n_top_words) ] )
    print()
    return topic_words, word_weights

def get_tfidf(abstracts, tokenizer):
    n_features = 5000 # 10000 # 5000 # 10000 # tf idf features 
    ng = 1 
    n_top_words = 20

    #n_components= 8 ## num topic
    
    ##======================
    print("Extracting tf-idf features for NMF...")
    tfidf_vectorizer = TfidfVectorizer(
        max_df=0.95,
        min_df=2,
        max_features=n_features,
        ngram_range=(ng,ng),
        tokenizer=tokenizer,
        stop_words="english",
        analyzer="word"
        )
    t0 = time()
    tfidf = tfidf_vectorizer.fit_transform(abstracts)
    vectorizer = tfidf_vectorizer
    print("tfidf fitting done in %0.3fs." % (time() - t0))
    return tfidf, vectorizer 

def NMF_topic_model(abstracts, tokenizer, 
                    tfidf=None, 
                    vectorizer=None, 
                    n_components= 15,
                    coherence='u_mass', #'u_mass',
                    abstracts_clean=None,
                    common_dictionary = None,
                    common_corpus = None,
                    return_coref = True,
                    return_infer = True,
                    ):
    ## define the hypermeters here, hard code 
    
    if tfidf is None:
        tfidf, vectorizer = get_tfidf(abstracts, tokenizer)

    t0 = time()
    alpha=0.1       ## hyper should adjust
    model = NMF(
        n_components=n_components, 
        random_state=1,
        alpha=alpha, 
        l1_ratio=.1, 
        verbose=False, #True,
        init='nndsvd', 
        max_iter=200,
        beta_loss="frobenius",           # default frobenius  "kullback-leibler",
        ).fit(tfidf)
    #dtm = csr_matrix(model.transform(tfidf))
    #components = csr_matrix(model.components_)
    print("nmf model fitting done in %0.3fs." % (time() - t0))

    tfidf_feature_names = vectorizer.get_feature_names() 
    # vectorizer.get_feature_names_out() 
    usable_topics, word_weights = show_top_words(
        model, tfidf_feature_names, 20, "Topics in NMF model (Frobenius norm)"
        )

    if return_infer:
        """
            num_doc * num_words = ( num_doc * num_topic ) * ( num_topic * num_words )
            topic prefenrence score  
        """
        doc2topic = model.transform( tfidf )
        topic_preference_score = np.sum( doc2topic, axis=0 ) 
        rel_topic_preference_score = topic_preference_score / np.mean(topic_preference_score) 
            ## measure how many words are related to the doc 
        topic2words = model.components_ 
        out_info = {
            "word_dic": tfidf_feature_names,
            "doc2topic": doc2topic.tolist(),
            "topic2words": topic2words.tolist(),
            "topic_preference_score" : topic_preference_score.tolist(),
            "rel_topic_preference_score": rel_topic_preference_score.tolist(),
            "topic_key_words": usable_topics,
            "topic_word_weights": word_weights, 

        } 
        return vectorizer, model, out_info 

    return usable_topics

    """
    ### the code below is used to calclate the c_v coherence 
    if common_dictionary is None or common_corpus is None: 
        abstracts_clean = [] 
        for text in tqdm( abstracts ):
            abstracts_clean.append( [word for word in tokenizer(text) ]  )
        #print("time of preprocess: ", time()-start); start = time()  
        common_dictionary = Dictionary(abstracts_clean)
        common_corpus = [common_dictionary.doc2bow(text) for text in abstracts_clean]
    if coherence=='c_v' :
        cm = CoherenceModel(
            topics=usable_topics,  
            texts=abstracts_clean, 
            dictionary=common_dictionary, 
            coherence='c_v')
    else:
        cm = CoherenceModel(
            topics=usable_topics, 
            corpus=common_corpus, 
            dictionary=common_dictionary, 
            coherence=coherence
            )
    #u_mass = cm.get_coherence_per_topic()
    coherence = cm.get_coherence() 
    return coherence 
    """


def LDA_topic_model(abstracts, tokenizer , n_components= 8):
    ## define the hypermeters here, hard code 
    n_features = 5000 # tf idf features 
        ## 10000 meets OOM problems
    ng = 1 
    n_top_words = 20

    #n_components=16 ## num topic
    alpha=0.1
    ##======================

    print("Extracting tf features for LDA...")
    tf_vectorizer = CountVectorizer(
        max_df=0.95,
        min_df=2,
        max_features=n_features,
        ngram_range=(ng,ng),
        tokenizer=tokenizer,
        stop_words="english",
        analyzer="word"
        )
    t0 = time()
    tf = tf_vectorizer.fit_transform(abstracts)
    vectorizer = tf_vectorizer
    print("tf fitting done in %0.3fs." % (time() - t0))

    t0 = time()
    model = LDA(
        n_components=n_components,
        doc_topic_prior=None,   # None = 1 / n_components   #n_components/50,
        max_iter=200,
        learning_method="online",
        n_jobs=2,
        learning_offset=50.0,
        )
    model.fit(tf)
    #dtm = csr_matrix(model.transform(tfidf))
    #components = csr_matrix(model.components_)
    print("LDA model fitting done in %0.3fs." % (time() - t0))

    # vectorizer.get_feature_names_out()
    usable_topics, _ = show_top_words(
        model, vectorizer.get_feature_names() , n_top_words, "Topics in LDA model with tf features"
        )
    return 

def text_process(text):
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(" +", " ", text).strip()
    return text 

def get_contexts(csv_fp):
    data = pd.read_csv( csv_fp )
    #abstracts = data['Abstract'].tolist()
    abstracts = [ text_process( txt1.strip() + " " + txt2.strip()) for txt1, txt2 in zip(data['Title'].tolist(),  data['Abstract'].tolist() )] 
    return abstracts 

def bert_topic_model(docs, tokenizer, num_topic):
    from bertopic import BERTopic
    st=time()
    topic_model = BERTopic()
    n_gram_range=(1,1)

    seed_topic_list = [ 
        ["disaster","earthquake","flood","storm","hurricane"], 
        ["health", "mental", "physical","ptsd","disease"] ,
        ["intervention","support","capital","resillience","network"],
        ] 
    topic_model = BERTopic(
        #embedding_model="all-MiniLM-L6-v2",
        language = "english",
        top_n_words = 10,           # default =10
        n_gram_range = n_gram_range,       # default = (1,1) 
            ## The n-gram range for the CountVectorizer
        min_topic_size = 10,
            # default 10, The minimum size of the topic. 
            # Increasing this value will lead
            # to a lower number of clusters/topics
        nr_topics = num_topic,
            ## num of topics
        low_memory = False,
        calculate_probabilities = False,
            #Whether to calculate the probabilities of all topics
            #per document instead of the probability of the assigned
            #topic per document.
        #diversity: float = None,
        seed_topic_list = seed_topic_list,
            # List[List[str]] = None,
            ##  A list of seed words per topic to converge around
        verbose = False,
        vectorizer_model = CountVectorizer(
            ngram_range=n_gram_range, 
            max_df=0.95,
            min_df=2,
            tokenizer=tokenizer,
            stop_words="english",
            analyzer="word"
            )
        )

    topics, probs = topic_model.fit_transform(docs)
    #new_topics, new_probs = topic_model.reduce_topics(docs, topics, probs, nr_topics=30) 
     
    for topic_idx in range(num_topic): 
        print("\ntopic %d"%(topic_idx))
        show = [ "[%s %.3f]"%(lst[0], lst[1]) for lst in topic_model.get_topic(topic=topic_idx) ]
        print("\t" + " ".join(show))
    print()
    print("process time : ", time()-st)
    import pdb; pdb.set_trace()
    out_model_fp="/home/linyu.linyu/health/params/bert_topic.bin"
    topic_model.save(out_model_fp)

    ## BERTopic.load(out_model_fp)
    ## BERTopic.find_topics("vehicle")

def guided_LDA(abstracts, tokenizer, n_components):
    ## https://www.kaggle.com/code/nvpsani/topic-modelling-using-guided-lda/notebook
    ## https://guidedlda.readthedocs.io/en/latest/
    
    import  guidedlda as glda
    n_features = 5000 # 10000
        ## 9391 words in total
    ng=1 
    seed_topic_list = [ 
        ["disaster","earthquake","flood","storm","hurricane"], 
        ["health", "mental", "physical","ptsd","disease"] ,
        ["intervention","support","capital","resillience","network"],
        ] 
    print("Extracting tf features for guild LDA...")
    tf_vectorizer = CountVectorizer(
        max_df=0.95,
        min_df=2,
        max_features=n_features,
        ngram_range=(ng,ng),
        tokenizer=tokenizer,
        stop_words="english",
        analyzer="word"
        )
    t0 = time()
    tf = tf_vectorizer.fit_transform(abstracts)
    vectorizer = tf_vectorizer
    print("tfidf fitting done in %0.3fs." % (time() - t0))
        ### cost 114s 

    t0 = time() 
    model = glda.GuidedLDA(
        n_topics=10, 
        n_iter=2000, 
        random_state=7, 
        refresh=20,
        alpha=0.01,
        eta=0.01
        )    
    model.fit(tf, 
        seed_topics=seed_topic_list, 
        seed_confidence=0.9, #0.15
    )
    print("glda fitting done in %0.3fs." % (time() - t0))
    ### about 100s 
  
    n_top_words = 10
    topic_word = model.topic_word_
    vocab = tf_vectorizer.vocabulary_ 
    rev_vocab = {v:k for k,v in vocab.items() }
    #import pdb; pdb.set_trace() 
    for i, topic_dist in enumerate(topic_word):
        topic_words = [ rev_vocab[idx] for idx in np.argsort(topic_dist)[:-(n_top_words+1):-1] ] 
        #np.argsort(topic_dist)[:-(n_top_words+1):-1]
        #np.array(vocab)
        print('Topic {}: {}'.format(i, ' '.join(topic_words)))


    return 


def main(csv_fp, model = "nmf"):
    n_components= 15

    abstracts =  get_contexts(csv_fp)
    tokenizer = get_tokenizer()
    if model == "nmf":
        NMF_topic_model(abstracts, tokenizer , n_components=n_components, return_coref=False) 
    elif model == "lda":
        LDA_topic_model(abstracts, tokenizer, n_components )
    elif model == "bert":
        ## run in torch env
        bert_topic_model(abstracts, tokenizer, n_components)
    elif model == "guildlda":
        ## run in torch env
        guided_LDA(abstracts, tokenizer, n_components)
    elif model == "gensim_nmf":
        gensim_NMF_model(abstracts, tokenizer) 
    return  

"""
        example to use LDA model in gensim
        https://towardsdatascience.com/evaluate-topic-model-in-python-latent-dirichlet-allocation-lda-7d57484bb5d0
"""

def gensim_NMF_model(
    abstracts, 
    tokenizer, 
    num_topics = 10,
    coherence='c_v',
    abstracts_clean=None, 
    common_dictionary=None, 
    common_corpus=None, 
    ):
    import gensim
    from gensim.utils import simple_preprocess
    from gensim.models import CoherenceModel, Nmf
    from gensim import corpora, models, similarities
    from gensim.models.ldamodel import LdaModel
    from gensim.corpora.dictionary import Dictionary


    start =time() 
    
    #num_topics = 10
    if common_dictionary is None or common_corpus is None: 
        abstracts_clean = [] 
        for text in tqdm( abstracts ):
            abstracts_clean.append( [word for word in tokenizer(text) ]  )
        print("time of preprocess: ", time()-start); start = time()  
        common_dictionary = Dictionary(abstracts_clean)
        common_corpus = [common_dictionary.doc2bow(text) for text in abstracts_clean]
    
    """
    model = Nmf( common_corpus, num_topics=num_topics )
    print("time of nmf model: ", time()-start); start = time()  
    coherence_model = CoherenceModel(model=model, texts=abstracts_clean, dictionary=common_dictionary, coherence='c_v')
    """

    model = LdaModel(common_corpus, num_topics, common_dictionary)
    coherence_model = CoherenceModel(model=model, corpus=common_corpus, coherence='u_mass') 
    coherence = coherence_model.get_coherence()
    print('topic %d Coherence Score %.3f '%( num_topics, coherence) )
    #import pdb; pdb.set_trace()     
    # Compute Coherence Score
    #import pdb; pdb.set_trace()
    return coherence

## manually calculate nmf model  use w2v model, could be used for different resource model
## https://github.com/derekgreene/topic-model-tutorial/blob/master/3%20-%20Parameter%20Selection%20for%20NMF.ipynb
## topic model in sklean : http://derekgreene.com/slides/topic-modelling-with-scikitlearn.pdf

def calculate_coherence( w2v_model, 
    term_rankings, 
    word_used=10,
    return_all=False
    ):
    """ 
        term_rankings: top k words for each topic  
        pretrained word2vec model in gensim
        word2vec-google-news-300 (1662 MB) (dimensionality: 300)
        word2vec-ruscorpora-300 (198 MB) (dimensionality: 300)
    """
    from itertools import combinations

    overall_coherence = 0.0
    all_topic_coherences = list()
    for topic_index in range(len(term_rankings)):
        # check each pair of terms
        pair_scores = []
    
        for pair in combinations( term_rankings[topic_index][:word_used], 2 ):
            try:
                #pair_scores.append( w2v_model.wv.similarity(pair[0], pair[1]) )
                pair_scores.append( w2v_model.similarity(pair[0], pair[1]) )
            except:
                print(pair)
                continue 
                import pdb; pdb.set_trace()
        # get the mean for all pairs in this topic
        topic_score = sum(pair_scores) / (len(pair_scores) + 1e-10)
        overall_coherence += topic_score
        all_topic_coherences.append( topic_score )
    # get the mean score across all topics
    if return_all:
        return all_topic_coherences 
    return overall_coherence / len(term_rankings)

def evaludate_nmf():
    ## https://zhuanlan.zhihu.com/p/75484791
    ##  
    cof_record = dict()
    abstracts =  get_contexts(csv_fp)
    tokenizer = get_tokenizer()
    
    common_dictionary = common_corpus = abstracts_clean = None 
    """
    abstracts_clean = [] 
    for text in tqdm( abstracts ):
        abstracts_clean.append( [word for word in tokenizer(text) ]  )
    #print("time of preprocess: ", time()-start); start = time()  
    common_dictionary = Dictionary(abstracts_clean)
    common_corpus = [common_dictionary.doc2bow(text) for text in abstracts_clean]
    """
    tfidf, vectorizer = get_tfidf(abstracts, tokenizer)
    ## *** ValueError: ("'texts' should be provided for %s coherence.", 'c_v')
    
    for n_components in range(16, 17):
        coherence = NMF_topic_model( abstracts, tokenizer,  tfidf, vectorizer, n_components, 'c_v', abstracts_clean,  common_dictionary,  common_corpus) 
        #coherence = gensim_NMF_model( abstracts, tokenizer,  n_components, 'c_v', abstracts_clean,  common_dictionary,  common_corpus)
        cof_record[ n_components ] = coherence
        print(n_components, coherence) 
    coref_scores = [  [cof_record[k], k ] for k in cof_record ]
    coref_scores.sort( reverse=True ) 
    for line in coref_scores: 
        print(line) 
    #import pdb; pdb.set_trace()


def evaludate_nmf_w2v(csv_fp):

    abstracts =  get_contexts(csv_fp)
    tokenizer = get_tokenizer()

    start = time()
    model_file = "/home/linyu.linyu/health/w2vmodel/gensim_glove.6B.300d.txt" 
    w2v_model = KeyedVectors.load_word2vec_format(model_file, binary=False)
    print("time to load w2v model: ", time()-start)
    # about 3 mins
    
    cof_record = {"5words": dict(), "10words": dict(), "20words": dict() }
    
    common_dictionary = common_corpus = None 
    tfidf, vectorizer = get_tfidf(abstracts, tokenizer)
    #for n_components in range(10, 45, 5):
    for n_components in range(10, 100, 5):  
        usable_topics = NMF_topic_model( abstracts, tokenizer,  tfidf, vectorizer, n_components,  return_coref = False) 
        coherence_10 = calculate_coherence( w2v_model, usable_topics, word_used=10)
        coherence_5 = calculate_coherence( w2v_model, usable_topics, word_used=5)
        coherence_20 = calculate_coherence( w2v_model, usable_topics, word_used=20)
         
        cof_record["5words"][ n_components ] = float(coherence_5)
        cof_record["10words"][ n_components ] = float( coherence_10 )
        cof_record["20words"][ n_components ] = float( coherence_20 )
        #print(n_components, coherence) 
    
    #import pdb; pdb.set_trace()
    out_fp = "/home/linyu.linyu/health/data/cls_output/topic_out/evaluate_num_topic_feature1w.json"
    out_fp = "/home/linyu.linyu/health/data/cls_output/topic_out/evaluate_num_topic_feature5k.json"
    out_fp = "/home/linyu.linyu/health/data/cls_output/topic_out/evaluate_num_topic_feature5k_20words.json"
      
    write_data( [json.dumps(cof_record)], out_fp )
    """
    coref_scores = [  [cof_record[k], k ] for k in cof_record ]
    coref_scores.sort( reverse=True ) 
    for line in coref_scores: 
        print(line) 
    #import pdb; pdb.set_trace()
    """

def write_data(out_data, out_fp ):
    if os.path.exists(out_fp):
        os.remove( out_fp )
    with open( out_fp, "w") as f :
        f.write("\n".join(out_data) )
    print(out_fp)
    return 


import joblib 
def write_model(model ,fp):
    if os.path.exists(fp):
        os.remove(fp)
    joblib.dump(model, fp) 
    return 

def infer_topic_model(csv_fp):
    abstracts =  get_contexts(csv_fp) 
    tokenizer = get_tokenizer()
    n_components= 40

    tfidf_model, nmf_model , out_json = NMF_topic_model(
        abstracts, tokenizer, 
        n_components = n_components, 
        return_infer = True,
        )
    
    #import pdb; pdb.set_trace() 
    out_fp = "/home/linyu.linyu/health/data/cls_output/topic_out/topic40_output.json"
    write_data( [json.dumps(out_json)], out_fp)
    
    model_folder = "/home/linyu.linyu/health/data/cls_output/model_out/"
    model_fp = model_folder  + "/topic.model"
    write_model(nmf_model ,model_fp) 
    #import pdb; pdb.set_trace() 
    """
    tfidf_fp = model_folder  + "/tfidf.model"
    write_model(tfidf_model ,tfidf_fp)
    loaded_model = joblib.load(model_fp)
    """
    return 


if __name__ == "__main__":
   
    
    csv_fp = "csv file path"

    #main(csv_fp, model = "nmf") ## runs about 3 s
    #main(csv_fp, model = "lda") ## runs about 125 s
    #main(csv_fp, model = "guildlda")    
    #main(csv_fp, model = "gensim_nmf" )
    #evaludate_nmf()
    
    ## use w2v coherence to determine nums of topic
    #evaludate_nmf_w2v(csv_fp)

    infer_topic_model(csv_fp)