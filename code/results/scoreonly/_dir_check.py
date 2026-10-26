import pandas as pd, numpy as np
so, ne = [], []
for s in range(5):
    f_so = rf'results/scoreonly/ecoli_P2_ScoreOnly_30_{s}_0_0.0_50.0_1.0.csv'
    f_ne = rf'results/shards/ecoli_P2_NoExploration_30_{s}_0_0.0_50.0_1.0.csv'
    a = pd.read_csv(f_so); b = pd.read_csv(f_ne)
    so.append(float(a.loc[np.isclose(a.budget, 0.10), 'ACC'].iloc[0]))
    ne.append(float(b.loc[np.isclose(b.budget, 0.10), 'ACC'].iloc[0]))
so = np.array(so); ne = np.array(ne); diff = (so - ne) * 100
print('seed   ScoreOnly  NoExpl   diff(pp)')
for i in range(5):
    print(f'{i:4d}  {so[i]:.4f}    {ne[i]:.4f}   {diff[i]:+.2f}')
print(f'mean ScoreOnly={so.mean():.4f}  mean NoExpl={ne.mean():.4f}  mean diff={diff.mean():+.2f} pp')
