[33;20m2022-04-28 00:41:19,306 - util.base_logger - WARNING - ./out already exists (utils.py:25)[0m
[32m2022-04-28 00:41:19,309 - util.base_logger - INFO - Created folder ./out/cluster_30-28-04-2022-00-41-19 (utils.py:28)[0m
[32m2022-04-28 00:41:19,310 - util.base_logger - INFO - Program starting in cluster-mode (main.py:64)[0m
[32m2022-04-28 00:41:19,310 - util.base_logger - INFO - STARTING PROGRAM (main.py:73)[0m
[32m2022-04-28 00:41:19,310 - util.base_logger - INFO - Selected parameters: (main.py:74)[0m
[32m2022-04-28 00:41:19,310 - util.base_logger - INFO - {'name': 'cluster_30-28-04-2022-00-41-19', 'train': True, 'model': './models/models-autoencodergray-scale.pth', 'letters': ['alpha', 'beta', 'chi', 'delta', 'epsilon', 'eta', 'gamma', 'iota', 'kappa', 'lambda', 'mu', 'nu', 'omega', 'omicron', 'phi', 'pi', 'psi', 'rho', 'sigma', 'tau', 'theta', 'xi', 'ypsilon', 'zeta'], 'logging': 40, 'mode': 'cluster', 'epochs': 30, 'dimensions': 28, 'tqdm': False, 'processing': 'gray-scale', 'root': 'out/cluster_30-28-04-2022-00-41-19'} (main.py:75)[0m
Traceback (most recent call last):
  File "main.py", line 96, in <module>
    X, y = autoencoder.train(run)
  File "/scicore/home/dokman0000/jabjan00/cvae-papyri/autoencoders/autoencoder.py", line 180, in train
    idx = [i for i in range(len(test_set)) if
  File "/scicore/home/dokman0000/jabjan00/cvae-papyri/autoencoders/autoencoder.py", line 181, in <listcomp>
    test_set.imgs[i][1] in [test_set.class_to_idx[letter] for letter in run.letters]]
  File "/scicore/home/dokman0000/jabjan00/cvae-papyri/autoencoders/autoencoder.py", line 181, in <listcomp>
    test_set.imgs[i][1] in [test_set.class_to_idx[letter] for letter in run.letters]]
KeyError: 'omicron'
