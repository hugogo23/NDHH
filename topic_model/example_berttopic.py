
"""
    https://github.com/MaartenGr/BERTopic
"""


from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups
from time import time  

def bert_topic_model(docs)
    
    st=time()
    #topic_model = BERTopic()
    topic_model = BERTopic(embedding_model="all-MiniLM-L6-v2")

    topics, probs = topic_model.fit_transform(docs)
    #print(topics)
    print(topic_model.get_topic_info())
    print("process time : ", time()-st)
    import pdb; pdb.set_trace()




#docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']
## docs size 18846, runs about 557s in cpu 

## q1: how the model do the sentenize ?
## q2: how is the model like ?