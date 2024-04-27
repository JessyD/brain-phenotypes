import numpy as np


rng = np.random.default_rng(230980234)

nsims = 100
ndraws = 1000
pop_size = 100000
pop_idxs = range(pop_size)
meta_res = []
for ix, min_r2 in enumerate(np.linspace(0.2, 0.3, 21)):
    min_r = np.sqrt(min_r2)
    r = [
        [min_r, 1],
        [1, min_r]
    ]
    res = []
    for simn in range(nsims):
        pop = rng.multivariate_normal([0, 0], r, size=pop_size)
        sim_res = []
        for drawn in range(ndraws):
            draw = rng.choice(pop, n, axis=0, replace=False)
            sim_res.append(np.corrcoef(draw[:, 0], draw[:, 1])[1,0])
        res.append(np.quantile(sim_res, 0.0275))
    res = np.array(res)
    meta_res.append(((res ** 2) > 0.2).mean())
    print(ix, end=',')
