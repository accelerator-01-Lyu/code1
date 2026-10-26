import sys, os, collections
sys.path.insert(0, r'D:\AL_FuzzyClustering_Paper\v2_unified\code')
import additional_common as ac
import pandas as pd

print('RANKED', ac.RANKED_METHODS)
d = os.path.join(ac.HERE, 'results', 'shards')
fs = [f for f in os.listdir(d) if f.endswith('.csv') and '_' in f]
print('n_files', len(fs))
print('seed_set counts:', collections.Counter(f.split('_')[2] for f in fs))
# sample one file
sample = [f for f in fs if f.split('_')[2] == '30'][:3]
for f in sample:
    print('SAMPLE', f)
df = pd.read_csv(os.path.join(d, sample[0]))
print('cols', list(df.columns))
print(df.head(3).to_string())
print('budgets', sorted(df['budget'].unique()) if 'budget' in df else 'NA')
