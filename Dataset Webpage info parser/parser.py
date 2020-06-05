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