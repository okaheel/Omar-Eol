import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


datasets = pd.read_csv('datasets-full.csv', error_bad_lines=False, low_memory=False, warn_bad_lines=False)
print("read data")

tfidf = TfidfVectorizer(stop_words='english')
datasets['title'] = datasets['title'].fillna('')
print("filled NA")

tfidf_matrix = tfidf.fit_transform(datasets['title'])
print("built matrix")

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
print(cosine_sim.shape)
print("Compute the cosine similarity matrix")

indices = pd.Series(datasets.index, index=datasets['title']).drop_duplicates()
#print(indices[:10])

def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]

    #get pairwise list of similarity scores with that title
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[0], reverse=True)
    sim_scores = sim_scores[1:11]
    dataset_indicies = [i[0] for i in sim_scores]
    return datasets['title'].iloc[dataset_indicies]

recommendations = get_recommendations('WY King Air Cloud Radar Flight Track Imagery [UWY]')
print(recommendations)
recommendations.to_csv('results.csv')