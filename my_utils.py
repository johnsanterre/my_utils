from __future__ import division
import numpy as np
#import matplotlib.pyplot as plt
#matplotlib.use('qt4agg')
#plt.style.use('ggplot')

import csv

from itertools import product
from collections import Counter
import datetime

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import accuracy_score, auc, f1_score, precision_score, recall_score, roc_curve
from sklearn.linear_model import LinearRegression, Lasso
from scipy.stats import rankdata
import difflib
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import networkx as nx


#supress Scientific Notation
np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=300)
np.set_printoptions(edgeitems=100)

def generate_rf_trees(data, labels, n_estimators=10, n_jobs=5):
  clf = RandomForestClassifier(n_estimators=n_estimators)
  clf.fit(data, labels)
  return [est.tree_.compute_feature_importances() for est in clf.estimators_]

def reverse_rank(x):
  return (len(x)+1) - rankdata(x).astype(int)

def convert_real_value_matrix_to_row_ranking_matrix(M):
  rM = np.zeros(shape=M.shape)
  for idx,x in enumerate(M):
    rM[idx] = reverse_rank(x)
  return rM

def percent_nonzero(M):
  return  len(np.nonzero(M)[0])/float(np.multiply(*M.shape))

#!import code; code.interact(local=vars())

def range_datetime(base, numdays):
    return [base - datetime.timedelta(days=x) for x in range(0, numdays)]

def save_session(floc='history.txt'):
    #overrights pevious history
    import readline
    readline.write_history_file(floc)

def get_tree_assoicated_with_feature(data, featureid):
  return np.nonzero(data[featureid])[0]

def get_postion_of_feature_in_trees(data, featureid):
  ret = []
  for x in get_tree_assoicated_with_feature(data,featureid):
    all_features =  np.nonzero(data[:,x])[0]
    vals = data[all_features, x]
    ranks = reverse_rank(vals)
    ret.append(ranks[np.where(all_features==featureid)])
  return ret

def convert_ordered_ranks_HR(sss):
    return [(x,rank_agg_idx[y]) for x, y in sss]

def setup_for_HR(clf_on_mM, M, labels):
    mM = remove_duplicate_columns(M)
    clf_on_mM.fit(mM,labels)
    tM = feature_matrix_from_clf(clf)
    all_idx = list(set([y for z in tM for x in  np.nonzero(z) for y in x]))
    all_idx.sort()
    all_idx=dict((idx,x) for (idx,x) in enumerate(all_idx)}
    return clf_on_mM, mM, tM

def rank_index_value_tuples(s):
    rank_index= sorted(zip((len(s)+1-rankdata(s)), range(len(s))), key=lambda x: x[0])
    ret = {}
    for idx,x in enumerate(rank_index):
      ret[x[0]] ={'loc':x[1], 'val':s[x[1]]}
    # {idx:{'loc':x, 'val':s[x]} for idx,x in enumerate(rank_index)}
    return ret

def get_Y_weight(tM):
    none_zero_list = list(set([y for z in tM for x in  np.nonzero(z) for y in x]))
    none_zero_list.sort()
    Y = np.zeros(shape=(len(none_zero_list),len(none_zero_list)))
    weight = np.zeros(shape=(len(none_zero_list),len(none_zero_list)))
    for i, row in enumerate(tM):
        locs = np.nonzero(row)[0]
        for combo in product(locs, repeat=2):
            zero = np.where(none_zero_list==combo[0])[0]
            one = np.where(none_zero_list==combo[1])[0]
            Y[zero,one] += row[combo[0]]-row[combo[1]]
            weight[zero,one] +=1
        print i
    return Y, weight

def hodgeRank(Y, weight):
    W = weight
    L = -W
    L = L + np.diag(sum(W))
    s = np.diag(np.dot(Y,W.T))
    return s
#transpose row/columns
#M[:,[0, 1]] = M[:,[1, 0]]

def get_top_features(clf, top_n):
    val = []
    loc = []
    test = clf.feature_importances_
    index = []
    for x in range(top_n):
        val.append(max(test))
        loc.append(np.where(test==max(test))[0][0])
        test = np.delete(test, loc[-1])
    for v in val:
        if v != 0:
            index.append(np.where(clf.feature_importances_==v)[0])

    idx = [x[0][0] for x in index] #test this
    return val, idx

#Numpy matrix manipuation funcitons
def matrix_to_binary_indicator_matrix(M):
    #example:
    return  (M>0).astype(int)

#def remove_duplicate_columns(M):
    #removes identical columns to shirnk the size of a matrix.
    #example:
#    return np.array(list({tuple(M[:,x]) for x in range(len(M[0]))})).T

def move_row_column_in_symetrical_matrix(M,a,b):
    #move row and column index a to row and column index b
    M[:,[a, b]] = M[:,[b, a]]
    M[[a, b],:] = M[[b, a],:]
    return M

def remove_all_zero_columns(M):
    #example:
    return M[:,np.nonzero(np.argmax(M, axis=0))[0]]

def get_max_val_by_row(M):
    return np.argmax(M, axis=1)

def get_max_val_by_column(M):
    return np.argmax(M, axis=0)

def distance_in_sequences(a,b):
    seq = difflib.SequenceMatcher(a=a, b=b)
    return seq.ratio()

def feature_matrix_from_clf(clf):
    #example:
    M = []
    for est in clf.estimators_:
        M.append(est.tree_.compute_feature_importances())
    return np.array(M)

#System level helpers
def pickle_this_datetime(obj, f_loc):
  import pickle
  end = str(datetime.datetime.now())
  f_loc = f_loc + '_' + end
  with open(f_loc, 'wb') as handle:
      pickle.dump(obj, handle)
  return

def pickle_this(obj, f_loc='cucumber.pickle'):
    import pickle
    with open(f_loc, 'wb') as handle:
        pickle.dump(obj, handle)
    return

def unpickle(f_loc='cucumber.pickle'):
    import pickle
    return pickle.load( open( f_loc,'rb'))

def run_bash(bashCommand):
    import subprocess
    process = subprocess.open(bashCommand.split())

def isfile(f_loc):
    import os.path
    return os.path.isfile(f_loc)

def import_file_from_url(f_loc='https://raw.githubusercontent.com/johnsanterre/my_utils/master/my_utils.py'):
    #snake eating itself
    import urllib2;
    tmp= urllib2.urlopen(f_loc);
    exec(tmp.read())

#Rick's group fucntions
def kmc_wrapper(k='-k25', m='-m8', f='-fm', origin='./', out='./NA.res', work='work-dir'):
    return ' '.join(['./kmc',k, m, f, origin, out, work])

def kmc_dump_wrapper(c='-cx0', origin='NA.res', out='./output',):
    return ' '.join(['./kmc_dump', c, origin, out])

def remove_singletons_from_M(M,index):
    singleton_locations = np.where(sum(M[:,]==1))[0]
    good_loc = list(set(range(M.shape[1]))-set(singleton_locations))
    inv_index = inverse_dict(index)
    M_out = np.zeros(shape=(M.shape[0], len(good_loc)))
    index_out = {}
    for idx, loc in enumerate(good_loc): M_out[:,idx]=M[:,loc]; index_out[idx]=inv_index[loc]
    return M_out, index_out

def make_labels(pos, neg):
    return np.concatenate((np.ones(neg),(np.ones(pos)+np.ones(pos))))

def read_fasta(floc):
    #assumes well structured, ie no error checking. h
    with open('all.fasta', 'rb') as f:
            reader = csv.reader(f,delimiter='\t')
            data = list(reader)
    all_fasta={}
    for row in data:
        if row[0][0]=='>':
            peg, func = row[0][1:].split(' ',1)
            all_fasta[peg]={'function':func}
            all_fasta[peg]={'protein':''}
        else:
            all_fasta[peg]['protein'] +=row[0]
    return row

def open_a_fangFang(loc, delim):
    with open(loc, 'rb') as f:
        reader = csv.reader(f, delimiter=delim)
        data = list(reader)
    index = {}
    inv_index = {}
    labels = data[0][1:]
    data = data[1:]

    for idx, x in enumerate(data):
        index[x[0]]=idx
        inv_index[idx]=x[0]

    data = np.transpose(np.array([x[1:] for x in data], dtype=(np.int16)))

    return data, labels, index, inv_index

def read_dataset(fname):
    #FF
    f = open(fname)
    cols = f.readline().split()
    skipcols = 1
    X = np.loadtxt(f, delimiter="\t", unpack=True, usecols=range(skipcols, len(cols)))
    y = map(lambda x: 1 if x == '1' else 0, cols[skipcols:])
    y = np.array(y)
    X_label = np.genfromtxt(fname, skip_header=1, delimiter="\t", usecols=[0], dtype='str')
    return X, y, X_label

#python
def inverse_dict(aDict):
    return dict((v, k) for (k, v) in aDict.items())
def rand_set(the_list, n_samples):
    import random
    random.shuffle(the_list)
    return the_list[0:n_samples]







#sklearn helpers
def run_clf(clf, skf, M, labels):
    acc= []
    for tr,tst in skf:
        clf.fit(M[tr],labels[tr])
        acc.append(accuracy_score(clf.predict(M[tst]),labels[tst]))
    return acc


#std format
def get_top_features_and_values(clf, n_top):
    return ( [ (x,clf.feature_importances_[x]) for
            x in clf.feature_importances_.argsort()[-n_top:][::-1]])

def top_important_features(clf, feature_names, num_features=20):
    #FF
    if not hasattr(clf, "feature_importances_"):
        return
    fi = clf.feature_importances_
    features = [ (f, n) for f, n in zip(fi, feature_names)]
    top = sorted(features, key=lambda f:f[0], reverse=True)[:num_features]
    return top

#exp helpers
def avg_acc_on_subset(clf,skf,M,labels,n):
    sub = rand_set(range(M.shape[0]),n)
    skf = StratifiedKFold(labels[sub], n_folds=5)
    return np.average(run_clf(clf,skf, M[sub,:],labels[sub]))

def avg_avg_acc_on_subset(clf, skf, M, labels, n, runs):
    acc = []
    for x in range(runs):
        acc.append(avg_acc_on_subset(clf,skf, M,labels,n))
    return np.average(acc), np.std(acc)

def avg_avg_acc_on_balanced_subset(clf,M,labels,n,runs):
    zeros = np.where(labels==0)
    ones = np.where(labels==1)
    acc=[]
    for x in range(runs):
        z = rand_set(zeros[0],n/2)
        o = rand_set(ones[0],n/2)
        t = list(z)+list(o)
        skf = StratifiedKFold(labels[t], n_folds=5)
        acc.append(np.average(run_clf(clf,skf,M[t,:],labels[t])))
    return np.average(acc), np.std(acc)

def run_subset_exp(clf, M, labels,ranges):
    results = []
    for r in ranges:
        tmp_results = [r]
        tmp_results.append(avg_avg_acc_on_balanced_subset(clf,M,labels,r,10))
        results.append(tmp_results)
    return results

# go thru this
def get_identicle_value_columns(M):
    col_ids = []
    for col_id, row in enumerate(M):
        value = row[0]
        if np.all(row==value, axis=1): # 1 for columns
            col_ids.append(col_id)
    return col_id
def agreement_fit(data,inv_index,type_one, type_two,count_Yes_No):
    one = [col_idx for col_idx in range(data.shape[1]) if np.count_nonzero(data[type_one, col_idx])==count_Yes_No[0]]
    two= [col_idx for col_idx in one if np.count_nonzero(data[type_two,col_idx])==count_Yes_No[1]]
    return [inv_index[x] for x in two]

def RES_id_from_SUS_id(SUS_ids, total_num_samples):
    return [x for x in range(total_num_samples) if x not in SUS_ids]

rf_sweep = {'clf': RandomForestClassifier,
             'n_estimators': [10,30,50],
             'max_features': ['sqrt','log2'],
             'max_depth': [None,4,7,15],
             'n_jobs':[1]}
ad_sweep = {'clf': AdaBoostClassifier, 'n_estimators': [20,50,100]}
#lr_sweep = {'clf': LogisticRegression, 'C': [1.0,2.0,0.5,0.25], 'penalty': ['l1','l2']}
#dc_tree = {'clf': DecisionTreeClassifier, 'max_depth': [None,4,7,15,25]},

std_clf= [rf_sweep]
cvs = [{'cv': StratifiedKFold}]
