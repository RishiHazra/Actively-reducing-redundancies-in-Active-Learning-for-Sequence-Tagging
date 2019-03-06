import collections
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

class KMedians(KMeans):

    def _e_step(self, X):
        self.labels_ = manhattan_distances(X, self.cluster_centers_).argmin(axis=1)

    def _average(self, X):
        return np.median(X, axis=0)


def word_tokenizer(text):
    #tokenizes and stems the text
    tokens = word_tokenize(text)
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(t) for t in tokens if t not in stopwords.words('english')]
    return tokens


def cluster_sentences(sentences, nb_of_clusters):
    tfidf_vectorizer = TfidfVectorizer(tokenizer=word_tokenizer,
                                    stop_words=stopwords.words('english'),
                                    max_df=0.9,
                                    min_df=0.1,
                                    lowercase=True)
    #builds a tf-idf matrix for the sentences
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
    print(tfidf_matrix.T.todense())
    x =tfidf_matrix.T.todense()
    #kmeans = KMeans(n_clusters=nb_of_clusters)
    kmedians = KMedians(k=3)
    kmedians.fit(tfidf_matrix)
    #kmeans.fit(tfidf_matrix)
    clusters = collections.defaultdict(list)
    for i, label in enumerate(kmedians.labels_):
            clusters[label].append(i)
    return dict(clusters),x




if __name__ == "__main__":
    sentences = ["Nature is beautiful","I like green apples",
            "We should protect the trees","Fruit trees provide fruits",
            "Green apples are tasty"]
    nclusters= 20
    clusters,x = cluster_sentences(sentences, nclusters)
    for cluster in range(nclusters):
            print("cluster ",cluster,":")
            for i,sentence in enumerate(clusters[cluster]):
                    print("\tsentence ",i,": ",sentences[sentence])