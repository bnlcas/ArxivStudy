import scipy.spatial.distance as dst
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import porter
import csv
from sklearn.decomposition import TruncatedSVD
import matplotlib.cm as cm
import string

def get_entropy(x):
    num_bins = np.ceil(np.sqrt(len(x)))
    bins = np.linspace(min(x),max(x), num_bins)
    bins[-1] += 1
    
    x = np.sort(x)
    p = np.zeros(len(bins)-1)
    for i in range(len(p)):
        tmp = (x >= bins[i])
        tmp2 = (x<bins[i+1])
        p[i] = np.sum(tmp*tmp2)
    p = p/np.sum(p)
    p = np.asarray([x for x in p if x != 0])
    
    H = np.sum(np.multiply(-p,np.log2(p)))
    return H




def diameter(X):
    # Takes a set of data points as rows and finds the diamter
    return float(max(dst.pdist(X)))
    
def separability(X,Y):
    # Finds the min distance between two sets of Vectors:
    sep_pt = []
    for i in range(np.shape(X)[0]):
        for j in range(np.shape(Y)[0]):
            sep_pt.append(float((dst.euclidean(Y[j,:],X[i,:]))))
    return min(sep_pt)

'''def dunn_sep_ind(X,c):
    #find the dunn separability index of cluster data:
    cats = list(set(c))
    diams = []
    for cat in cats:
        inds = [i for i in range(len(c)) if cat in c[i]]
        if len(inds) > 1:
            diams.append(diameter(X[inds,:]))
    seps = []
    for cat1_ind, cat1 in enumerate(cats[:-1]):
        inds1 = [i for i, cat in enumerate(c) if cat1 in cat]
        for cat2_ind, cat2 in enumerate(cats[(cat1_ind+1):]):
            inds2 = [i for i, cat enumerate(c) if cat2 in cat]
            if len(inds1)>1 and len(inds2)>1:
                seps.append(separability(X[inds1,:],X[inds2,:]))
    return float(max(diams))/(min(seps)+0.0001)'''
    
  
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
    
## Take list of abstracts, stem words and remove stop words:
stop_words= stopwords.words('english')
arxive_stop_words = ['results', 'paper','present', 'study', 'recent','discuss']
#stop_words.extend(topics)
stop_words.extend([''])



## Generate text, document matrix for LSA
stemmer = porter.PorterStemmer()
abstract_filtered = []
abstract_words = []
for abstract in abstracts:
    abstract = abstract.decode('utf-8')
    for p in string.punctuation:
        abstract = string.replace(abstract,p,' ')
    tmp = [stemmer.stem(word) for word in abstract.split(' ') if stemmer.stem(word) not in stop_words]
    abstract_filtered.append(tmp)
    abstract_words.extend(tmp)
    

abstract_words = sorted(abstract_words)
unique_words = set(abstract_words)
common_words = []
common_thresh = 3
'''
for word in unique_words:
    first_ind = next(x[0] for x in enumerate(abstract_words) if x[1] == word)
    if (abstract_words[first_ind:(first_ind+common_thresh)].count(word) == common_thresh):
        common_words.append(word)
'''

## ALTERNATE PROTOCOL
unique_words = sorted(list(unique_words))
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


    

#abstract_word_count = [abstract_words.count(word) for word in unique_words]
#sorted(zip(unique_words,abstract_word_count), key= lambda x: x[1], reverse=True)
'''
unique_words = set(abstract_words)
common_words = []
common_thresh = 3
for word in unique_words:
    if abstract_words.count(word) > common_thresh:
        common_words.append(word)
'''
## Generate Term Document Matrix
X = np.zeros((len(abstract_filtered),len(common_words)))
for i, abstract in enumerate(abstract_filtered):
    for word in set(abstract):
        word_count = abstract.count(word)
        try:
            ind = next(x[0] for x in enumerate(common_words) if x[1] == word)
            X[i,ind] = word_count
        except StopIteration:
            a = 1
            
start_inds = range(0,len(abstract_filtered),1000)
end_inds = range(1000,len(abstract_filtered),1000)
end_inds.extend([len(abstract_filtered)])
common_words_enum = enumerate(common_words)
for k in range(len(start_inds)):
    abstract_filtered_range = abstract_filtered[start_inds[k]:end_inds[k]]
    for j, word in common_words_enum :
        for i, abstract in enumerate(abstract_filtered_range):
            X[i+start_inds[k],j] = abstract.count(word)

# Clear zero_rows (there can be none)


## Process Term-Doc Mat
X_basic = np.log(X+1)
# Divide by Entropy:

for i in range(np.shape(X)[1]):
	col = X[:,i]
	prob = np.divide(col,np.sum(col))
	prob = prob[np.nonzero(prob)]
	entropy = -np.sum(np.multiply(prob, np.log2(prob)))
	X[:,i] = np.divide(col, entropy)
X = np.log(X+1)
'''	
for i in range(np.shape(X)[0])
    row = X[i,:]
    H = get_entropy(row)
    X[i,:] = np.divide(X[i,:],H)
    '''

root_dir = '/Volumes/NO NAME/Arxiv Study/'
np.save(root_dir+'Abstracts_Raw', X_basic)
np.save(root_dir+'Doc_Term_mat_Raw',X)

# X = np.load(root_dir +'Doc_Term_mat_RAW.npy')
ind = next(x[0] for x in enumerate(common_words) if x[1] == word)



primary_topics_main = map(lambda x: x.split('.')[0], primary_topics)
cat_size_thresh = 3
cats = list(set(primary_topics_main))
small_cats = [cat for cat in cats if primary_topics_main.count(cat) < cat_size_thresh]

small_cat_inds = []
for cat in small_cats:
    cat_inds = [i for i, item in enumerate(primary_topics_main) if item == cat]
    small_cat_inds.extend(cat_inds)
if len(small_cat_inds) > 0:
    primary_topics_main = [row for i, row in enumerate(primary_topics_main) if not i in small_cat_inds]
    X = np.delete(X,small_cat_inds,0)

    

##Run LSA - check number of components vs clustering:
comps = [2, 5, 10, 20, 50, 100, 200, 300, 500, 1000, 3000, 10000]
dunn_inds = []
for lsa_comps in comps:
    svd = TruncatedSVD(n_components=lsa_comps)
    X_svd = svd.fit_transform(X)
    dunn_ind = dunn_sep_ind(X_svd, primary_topics_main)
    dunn_inds.append(dunn_ind)
    
dunn_v_comps = zip(comps, dunn_inds)
with open(root_dir+'dunn_v_comps.csv', 'wb') as f:
    writer = csv.writer(f,)
    writer.wrierows(dunnv_comps)
    
## Get to ward stuff:
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import ward, dendrogram

svd = TruncatedSVD(n_components = 300)
X_svd = svd.fit_transform(X)

cats = list(set(primary_topics_main))

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
    inds = [i for i, item in enumerate(primary_topics_main) if item i in cat]
    centroids.append(np.mean(X_svd[inds,:],0))

Z = ward(centroids)

fig, ax = plt.subplots(figsize=(16, 11)) # set size
ax = dendrogram(Z, orientation="right", labels=cat_full_list, color_threshold = 2);

plt.tick_params(\
    axis= 'x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off')

plt.tight_layout() #show plot with tight layout


#plot mds
from sklearn.manifold import MDS
mds = MDS(n_components=2)
X_proj = mds.fit_transform(X_svd[1:1000,:])


cats_mds = list(set(primary_topics_main[1:1000]))
color_map = cm.rainbow(np.linspace(0, 1, len(cats_mds)))
#fig, ax = plt.subplots(figsize=(8, 8))
for x, cat in enumerate(cats_mds):
    cat_inds = [i for i, item in enumerate(primary_topics_main[1:1000]) if item in cat]
    plt.scatter(X_proj[cat_inds,0], X_proj[cat_inds,1], color = color_map[x], label = cat)
    plt.legend()
tmp = []
for item in cat_full:
    tmp.extend(item)
                     #color=cluster_colors[cat])
ax = sca
