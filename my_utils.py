from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.cross_validation import StratifiedKFold

low_ranges = [10,20,30,40,50,60,70,80,90,100]    
mid_ranges = [100,150,200,250,300,350,400,450,500]
high_ranges = [500,800, 1100,1400, 1700, 2000, 2300, 2600]
def import_file_from_url(f_loc='https://raw.githubusercontent.com/johnsanterre/my_utils/master/my_utils.py'):
    import urllib2; 
    tmp= urllib2.urlopen(f_loc);
    exec(tmp.read())

def pickle_this(obj, name='cucumber.pickle'):
    import pickle
    with open(name, 'wb') as handle:  pickle.dump(obj, handle)
    return

def unpickle(f_loc='cucumber.pickle'):
    import pickle    
    return pickle.load( open( f_loc,'rb'))

def run_bash(bashCommand):
    import subprocess
    process = subprocess.Popen(bashCommand.split())
    
def isfile(f_loc):
    import os.path
    return os.path.isfile(f_loc)

def kmc_wrapper(k='-k25', m='-m8', f='-fm', origin='./', out='./NA.res', work='work-dir'):
    return ' '.join(['./kmc',k, m, f, origin, out, work])

def kmc_dump_wrapper(c='-cx0', origin='NA.res', out='./output',):
    return ' '.join(['./kmc_dump', c, origin, out])


def get_top_features_and_values(clf, n_top):
        return ( [ (x,clf.feature_importances_[x]) for
                x in clf.feature_importances_.argsort()[-n_top:][::-1]])

def inverse_dict(aDict):
    return {v: k for k, v in aDict.items()}
        
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
    

# go thru this

def open_a_fangFang(loc, delim):
    import csv
    with open(loc, 'rb') as f:
        reader = csv.reader(f, delimiter=delim)
        data = list(reader)
    index = {}
    inv_index = {}
    for idx, x in enumerate(data):
        index[x[0]]=idx
        inv_index[idx]=x[0]
    data = np.transpose(np.array([x[1:] for x in data], dtype=(np.int16))) #removes featureID
    return data, index, inv_index

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
    
    
def rand_set(the_list, n_samples):
    import random
    random.shuffle(the_list)
    return the_list[0:n_samples]





def run_clf(clf, skf, M, labels):
    acc= []
    for tr,tst in skf: 
        clf.fit(M[tr],labels[tr])
        acc.append(accuracy_score(clf.predict(M[tst]),labels[tst]))
    return acc

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
    

    
    
    
    
