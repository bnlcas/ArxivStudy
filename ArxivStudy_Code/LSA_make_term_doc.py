import scipy.spatial.distance as dst
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import porter
import csv
from sklearn.decomposition import TruncatedSVD
import matplotlib.cm as cm
import string
from sklearn.feature_extraction.text import CountVectorizer


root_dir = '/Volumes/NO NAME/Arxiv Study/'

primary_topics = []
with open(root_dir+'Topic_List.csv','rb') as b:
    reader = csv.reader(b)
    for row in reader:
        primary_topics.append(''.join(row))
        
abstracts = []
with open(root_dir+'Abstract_List.csv','rb') as b:
    reader = csv.reader(b)
    for row in reader:
        abstracts.append(''.join(row))
abstracts = [string.replace(a,'\n', ' ') for a in abstracts]

## RESTRICT To smaller number per topic
max_topic_size = 400
abstracts_restrict = []
primary_topics_restrict = []
for topic in set(primary_topics):
    inds = [i for i, x in enumerate(primary_topics) if x == topic]
    if len(inds)>max_topic_size:
        inds = inds[0:max_topic_size]
    for i in inds:
        abstracts_restrict.append(abstracts[i])
        primary_topics_restrict.append(primary_topics[i])

abstracts = abstracts_restrict
primary_topics = primary_topics_restrict


## Take list of abstracts, stem words and remove stop words:
stop_words= stopwords.words('english')
arxive_stop_words = ['results', 'paper','present', 'study', 'recent','discuss']
#stop_words.extend(topics)
stop_words.extend([''])



## Generate text, document matrix for LSA
stemmer = porter.PorterStemmer()
abstracts_filtered = []
abstract_words = []
for abstract in abstracts:
    abstract = abstract.decode('utf-8')
    for p in string.punctuation:
        abstract = string.replace(abstract,p,' ')
    tmp = [stemmer.stem(word) for word in abstract.split(' ') if stemmer.stem(word) not in stop_words]
    abstracts_filtered.append(tmp)
    abstract_words.extend(tmp)
    
abstract_words = sorted(abstract_words)
unique_words = sorted(list(set(abstract_words)))

common_words = []
common_thresh = 3
unq_ind = 0
abstr_ind = 0

word = unique_words[unq_ind]
if abstract_words[abstr_ind :(abstr_ind +common_thresh)].count(word) == common_thresh:
    common_words.append(word)
unq_ind +=1



while unq_ind < len(unique_words) and abstr_ind < len(abstract_words):
    word = unique_words[unq_ind]
    while abstract_words[abstr_ind] != word:
        abstr_ind += 1
    if abstract_words[abstr_ind :(abstr_ind +common_thresh)].count(word) == common_thresh:
        common_words.append(word)
    unq_ind +=1


bad_inds = [i for i in range(len(abstracts_filtered)) if len(abstracts_filtered[i]) < 2]

abstracts_filtered = [x for i, x in enumerate(abstracts_filtered) if i not in bad_inds]
primary_topics = [x for i, x in enumerate(primary_topics) if i not in bad_inds]

abstract_filtered_str = [string.join(x, ' ') for x in abstracts_filtered]

vectorizer = CountVectorizer(analyzer = 'word', max_features = len(common_words))
vectorizer.fit(abstract_filtered_str)
vectorized_data = vectorizer.transform(abstract_filtered_str)
X = vectorized_data.todense()


##
# X_load = np.load(root_dir+'Abstracts_LSA_tdif.npy')

#X_basic = np.log(X+1)
# Divide by Entropy:
'''
for i in range(np.shape(X)[1]):
	col = X[:,i]
	if (np.sum(col) == 0):
	    X[:,i] = col
	else:
	    norm = np.sum(col)*1.0
    	prob = np.divide(col,norm)
    	prob = prob[np.nonzero(prob)]
    	entropy = 1-np.sum(np.multiply(prob, np.log2(prob)))
    	X[:,i] = np.divide(col, entropy)
    	'''
## Divide by number of occurance
'''for i in range(np.shape(X)[1]):
	col = X[:,i]
	if (np.sum(col) == 0):
	    X[:,i] = col
	else:
	    norm = np.sum(col)*1.0
	    X[:,i] = np.divide(col, norm)
	    '''
X = np.log(X+1)

svd = TruncatedSVD(n_components = 300)
X_svd = svd.fit_transform(X)

#X_basic_svd =  svd.fit_transform(X_basic)



## fit data to clustering
root_dir = '/Volumes/NO NAME/Arxiv Study/'
np.save(root_dir+'Abstracts_LSA_tdif', X_svd)



from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import ward, dendrogram
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from sklearn.neighbors import radius_neighbors_graph

cats = list(set(primary_topics))
X_blowup = X_svd
for i, topic in enumerate(cats):
	topic_inds = [j for j,x in enumerate(primary_topics) if x == topic]
	for j  in topic_inds:
		row = X_svd[j,:]
		X_blowup[j,:] = np.ones(len(row))*100.0*(i+1)
#connectivity_g = kneighbors_graph(X, n_neighbors = 50)
connectivity_g = radius_neighbors_graph(X_blowup, 50, mode='connectivity',include_self=True)


Z = AgglomerativeClustering(n_clusters=len(cats), connectivity=connectivity_g,
                               linkage='ward').fit(X_svd)



abrevs_list = []
with open(root_dir+'ArxivAbrevs.csv','rb') as b:
    reader = csv.reader(b)
    for row in reader:
        abrevs_list.append(row)
cat_full = [[item[1] for item in abrevs_list if cat == item[0]] for cat in cats]
cat_full_list = []
for item in cat_full:
    cat_full_list.extend(item)

centroids = []
for cat in cats:
    inds = [i for i, item in enumerate(primary_topics) if item in cat]
    centroids.append(np.mean(X_svd[inds,:],0))

#Z = ward(centroids)
Z = linkage(centroids,method = 'complete',metric='euclidean')

fig, ax = plt.subplots(figsize=(16, 13)) # set size
ax = dendrogram(Z, orientation="right", labels=cat_full_list, color_threshold = 0.3);

plt.tick_params(\
    axis= 'x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off')

plt.tight_layout() #show plot with tight layout



