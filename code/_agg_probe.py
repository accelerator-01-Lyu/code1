import run_experiments as R
X, y = R.load_pkl("aggregation")
R.run_traj(X, y, R.NCLASSES["aggregation"], "aggregation", "P2_Full",
           0, 0, 0.0, 50.0, 1.0, budgets=(0.1,))
print("done")
