from bs4 import BeautifulSoup
from urllib.request import urlopen
from nltk.tokenize import sent_tokenize
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

def get_only_text(url):
    """ 
    return text from dataset
    at the specified url
    """
    page = urlopen(url)
    soup = BeautifulSoup(page, "lxml")
    text = ' '.join(map(lambda p: p.text, soup.find_all('p')))
  
    print ("=====================")
    print (text)
    print ("=====================")
 
    return soup.title.text, text    
 
     
url="https://data.eol.ucar.edu/project/TORUS_2019"
text = get_only_text(url)


sentences = []
for s in text:
    sentences.append(sent_tokenize(s))

sentences = [y for x in sentences for y in x]
sentences[30:40]

# Extract word vectors
word_embeddings = {}
f = open('glove.6B.100d.txt', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    word_embeddings[word] = coefs
f.close()

len(word_embeddings)

# remove punctuations, numbers and special characters
clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")

# change to lowercase
clean_sentences = [s.lower() for s in clean_sentences]
stop_words = stopwords.words('english')
# function to remove stopwords
def remove_stopwords(sen):
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new
clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]

sentence_vectors = []
for i in clean_sentences:
    if len(i) != 0:
        v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
    else:
        v = np.zeros((100,))
    sentence_vectors.append(v)

# Create an empty similarity matrix
sim_mat = np.zeros([len(sentences), len(sentences)])

for i in range(len(sentences)):
    for j in range(len(sentences)):
        if i != j:
            sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]

nx_graph = nx.from_numpy_array(sim_mat)
scores = nx.pagerank(nx_graph)
ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
# Extract top 15 sentences as the summary representation
for i in range(10):
    print(ranked_sentences[i][1])


##Method 2
LANGUAGE = "english"
SENTENCES_COUNT = 10


parser = HtmlParser.from_url(url, Tokenizer(LANGUAGE))
print ("--LsaSummarizer--")    
summarizer = LsaSummarizer()
summarizer = LsaSummarizer(Stemmer(LANGUAGE))
summarizer.stop_words = get_stop_words(LANGUAGE)
for sentence in summarizer(parser.document, SENTENCES_COUNT):
    print(sentence)


print ("--LuhnSummarizer--")     
summarizer = LuhnSummarizer() 
summarizer = LuhnSummarizer(Stemmer(LANGUAGE))
#replace with terms to remove
summarizer.stop_words = ("I", "am", "the", "you", "are", "me", "is", "than", "that", "this")
for sentence in summarizer(parser.document, SENTENCES_COUNT):
    print(sentence)

print ("--EdmundsonSummarizer--")     
summarizer = EdmundsonSummarizer() 
#replace with terms to remove
words1 = ("economy", "fight", "trade", "china")
summarizer.bonus_words = words1

#replace with terms to remove
words2 = ("another", "and", "some", "next")
summarizer.stigma_words = words2
    
#replace with terms to remove
words3 = ("another", "and", "some", "next")
summarizer.null_words = words3
for sentence in summarizer(parser.document, SENTENCES_COUNT):
    print(sentence)

print ("--LexRankSummarizer--")   
summarizer = LexRankSummarizer()
summarizer = LexRankSummarizer(Stemmer(LANGUAGE))
summarizer.stop_words = ("I", "am", "the", "you", "are", "me", "is", "than", "that", "this")
for sentence in summarizer(parser.document, SENTENCES_COUNT):
    print(sentence)