from collections import Counter

#    lol -> list of lists
#    lod -> list of dicts
#    dod -> dic of dicts
#   dlod -> dic of lod
#      D -> data = i.e. undetermined type
#      s -> string

# assumption is that there are two basic data types a dlod and a dod.

tmp_lod = [dict(fake_key='f1', A='a1', B='b1'),
           dict(fake_key='f2', A='a2', B='b2'),
           dict(fake_key='f3', A='a3', B='b3'),
           dict(fake_key='f4', A='a4', B='b4')]

def search_types(D):
    # not tested
    # change to include type leaf nodes ?
    # assumes all entries are same type and depth is equal
    if type(D) in [list, dict, tuple]:
        if type(D) == dict:
            return str(type(D)) + search_types(D[D.keys()[0]])
        else:
            return str(type(D)) + search_types(D[0])
    return ''
def cast_to_short_cuts(s):
    # are these all reasonable types?
    if "<type 'list'><type 'dict'>" == s:
        return 'lod'
    elif "<type 'list'><type 'list'>" == s:
        return 'lol'
    elif "<type 'dict'><type 'list'>" == s:
        return 'dol'
    elif "<type 'dict'><type 'dict'>" == s:
        return 'dod'
    elif "<type 'dict'><type 'list'><type 'dict'>" == s:
        return 'dod'
    return 'error'
def check(D):
    return cast_to_short_cuts(search_types(D))
def unique_values_from_d_in_lod_by_key(lod, s, verbose=False):
    # test
    # type D?
    # lod = lod?
    # make a list of unique items
    v = set([d[s] for d in lod])
    if verbose:
        print "Total Number of Unique", s, "'s:", len(v)
    return v
def convert_lod_to_dlod_by_key(lod, key):
    values = unique_values_from_d_in_lod_by_key(lod, key)
    a_dict = {}
    for v in values:
        a_dict[v] = [x for x in lod if x[key] == v]
    return a_dict
def get(dod, key):
    # test
    # type check dod ?
    return [dod[v][key] for v in dod.keys()]
def cnt_by_key(dod, k):
    return Counter(get(dod,k))[True]
def print_dlod_subset_keys(dlod, k, header='name,desc,latitude,longitude',
                                    keys=['date','latitude','longitude']):
    if keys == None:
        keys = dlod.keys()
    ts = ",".join(['%s' for x in range(len(keys))])
    print header
    for entry in dlod[k]:
        print ts % tuple([str(entry[x]) for x in keys])
    return

def superset_of_keys(t):
    # takes lod, or dlod
    return
def superset_of_items_by_key(t):
    # takes lod, or dlod
    return
