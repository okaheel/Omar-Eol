raw_docs = ["The RELAMPAGO  field program will be conducted in west  central Argentina in the general vicinity of the Sierras de Cordoba (SDC) and the Andes foothills  near Mendoza. This region arguably has among the most intense convective systems in the world  with respect to the frequency of large hail, high storm tops, and extreme lightning activity,  yet much remains unknown about the scarcely observed intense convection in this region. RELAMPAGO,  leveraging the repeatability of storms in the region, aims to address science questions related  to the pre-initiation to initiation, initial organization/severe-weather generation, and growth/backbuilding  stages of storm development, which are poorly understood. New insights into sconnections between  the extreme hydroclimate, high impact weather, and atmospheric dynamical processes in meteorological  and geographical settings unique to the these regions can be obtained by bringing together  NSF facilities with (1) new operational dual-polarization radars in Argentina, (2) significant  contributions from Argentina, Brazil, Chile, NOAA, and NASA, and (3) a complementary funded  U.S. Department of Energy major field campaign called Clouds, Aerosols, and Complex Terrain  Interactions (CACTI)."]

# Tokenizing text into bags of words
from nltk.tokenize import word_tokenize
tokenized_docs = [word_tokenize(doc) for doc in raw_docs]
print(tokenized_docs)

# Removing punctuation
import re
import string
regex = re.compile('[%s]' % re.escape(string.punctuation))

tokenized_docs_no_punctuation = []

for review in tokenized_docs:
    new_review = []
    for token in review:
        new_token = regex.sub(u'', token)
        if not new_token == u'':
            new_review.append(new_token)
    
    tokenized_docs_no_punctuation.append(new_review)
    
print(tokenized_docs_no_punctuation)

# Cleaning text of stopwords
from nltk.corpus import stopwords

tokenized_docs_no_stopwords = []

for doc in tokenized_docs_no_punctuation:
    new_term_vector = []
    for word in doc:
        if not word in stopwords.words('english'):
            new_term_vector.append(word)
    
    tokenized_docs_no_stopwords.append(new_term_vector)

print(tokenized_docs_no_stopwords)

# Stemming and Lemmatizing
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

porter = PorterStemmer()
snowball = SnowballStemmer('english')
wordnet = WordNetLemmatizer()

preprocessed_docs = []

for doc in tokenized_docs_no_stopwords:
    final_doc = []
    for word in doc:
        final_doc.append(porter.stem(word))
        #final_doc.append(snowball.stem(word))
        #final_doc.append(wordnet.lemmatize(word))
    
    preprocessed_docs.append(final_doc)

print(preprocessed_docs)