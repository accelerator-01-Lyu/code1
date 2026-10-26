import pickle, os, numpy as np
d = r'D:\AL_FuzzyClustering_Paper\v2_unified\code\reference_repo\data'
for name in ['iris', 'wine', 'glass', 'ecoli', 'aggregation', 'compound']:
    o = pickle.load(open(os.path.join(d, name + '.pkl'), 'rb'))
    X, y = o['X'], o['y']
    print(f"{name:12s} X={X.shape} y={y.shape} classes={len(np.unique(y))} "
          f"info={o.get('info','')}")
