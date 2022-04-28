[33;20m2022-04-28 02:37:48,198 - util.base_logger - WARNING - ./out already exists (utils.py:25)[0m
[32m2022-04-28 02:37:48,204 - util.base_logger - INFO - Created folder ./out/cluster_100-28-04-2022-02-37-48 (utils.py:28)[0m
[32m2022-04-28 02:37:48,205 - util.base_logger - INFO - Program starting in cluster-mode (main.py:64)[0m
[32m2022-04-28 02:37:48,205 - util.base_logger - INFO - STARTING PROGRAM (main.py:73)[0m
[32m2022-04-28 02:37:48,206 - util.base_logger - INFO - Selected parameters: (main.py:74)[0m
[32m2022-04-28 02:37:48,206 - util.base_logger - INFO - {'name': 'cluster_100-28-04-2022-02-37-48', 'train': True, 'model': './models/models-autoencodergray-scale.pth', 'letters': ['alpha', 'beta', 'chi', 'delta', 'epsilon', 'eta', 'gamma', 'iota', 'kappa', 'lambda', 'mu', 'nu', 'omega', 'omicron', 'phi', 'pi', 'psi', 'rho', 'sigma', 'tau', 'theta', 'xi', 'ypsilon', 'zeta'], 'logging': 40, 'mode': 'cluster', 'epochs': 100, 'dimensions': 28, 'tqdm': False, 'processing': 'gray-scale', 'root': 'out/cluster_100-28-04-2022-02-37-48'} (main.py:75)[0m
[32m2022-04-28 02:37:48,404 - util.base_logger - INFO - Network(
  (layer1): Sequential(
    (0): Conv2d(1, 16, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (layer2): Sequential(
    (0): Conv2d(16, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (fc): Sequential(
    (0): Linear(in_features=1568, out_features=256, bias=True)
    (1): Dropout(p=0.2, inplace=False)
    (2): ReLU()
  )
  (out): Linear(in_features=256, out_features=5, bias=True)
) (autoencoder.py:193)[0m
[32m2022-04-28 02:37:48,406 - util.base_logger - INFO - ConvAutoEncoder(
  (encoder): Sequential(
    (0): Conv2d(1, 16, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (4): Conv2d(16, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (5): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): ReLU()
    (7): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (8): Conv2d(32, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (9): BatchNorm2d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (10): ReLU()
  )
  (decoder): Sequential(
    (0): Conv2d(4, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): Upsample(scale_factor=2.0, mode=nearest)
    (4): Conv2d(32, 16, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (5): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): ReLU()
    (7): Upsample(scale_factor=2.0, mode=nearest)
    (8): Conv2d(16, 1, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (9): BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (10): Sigmoid()
  )
) (autoencoder.py:199)[0m
[32m2022-04-28 03:27:43,682 - util.base_logger - INFO - Epoch=0 done. (autoencoder.py:252)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 03:28:42,002 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 03:28:42,009 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 03:28:42,009 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48 already exists (utils.py:25)[0m
[32m2022-04-28 03:28:42,010 - util.base_logger - INFO - Created folder ././out/cluster_100-28-04-2022-02-37-48/CovAE (utils.py:28)[0m
[32m2022-04-28 03:28:42,010 - util.base_logger - INFO - Created folder ././out/cluster_100-28-04-2022-02-37-48/CovAE/gray-scale (utils.py:28)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 03:29:06,105 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 03:29:06,105 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 03:29:06,105 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48 already exists (utils.py:25)[0m
[33;20m2022-04-28 03:29:06,105 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 03:29:06,105 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 03:52:36,183 - util.base_logger - INFO - Epoch=1 done. (autoencoder.py:252)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 03:53:36,281 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 03:53:36,288 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 03:53:36,288 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48 already exists (utils.py:25)[0m
[33;20m2022-04-28 03:53:36,288 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 03:53:36,288 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 03:53:49,288 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 03:53:49,288 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 03:53:49,288 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48 already exists (utils.py:25)[0m
[33;20m2022-04-28 03:53:49,288 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 03:53:49,288 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 04:17:11,945 - util.base_logger - INFO - Epoch=2 done. (autoencoder.py:252)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 04:18:11,215 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 04:18:11,216 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 04:18:11,216 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48 already exists (utils.py:25)[0m
[33;20m2022-04-28 04:18:11,216 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 04:18:11,216 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 04:18:27,823 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 04:18:27,823 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 04:18:27,823 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48 already exists (utils.py:25)[0m
[33;20m2022-04-28 04:18:27,823 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 04:18:27,823 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 04:41:51,229 - util.base_logger - INFO - Epoch=3 done. (autoencoder.py:252)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 04:42:48,095 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 04:42:48,097 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 04:42:48,097 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48 already exists (utils.py:25)[0m
[33;20m2022-04-28 04:42:48,097 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 04:42:48,097 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 04:43:01,114 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 04:43:01,114 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 04:43:01,114 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48 already exists (utils.py:25)[0m
[33;20m2022-04-28 04:43:01,114 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 04:43:01,114 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 05:06:28,157 - util.base_logger - INFO - Epoch=4 done. (autoencoder.py:252)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 05:07:27,840 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 05:07:27,840 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 05:07:27,840 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48 already exists (utils.py:25)[0m
[33;20m2022-04-28 05:07:27,840 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 05:07:27,840 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 05:07:41,113 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 05:07:41,114 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 05:07:41,114 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48 already exists (utils.py:25)[0m
[33;20m2022-04-28 05:07:41,114 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 05:07:41,114 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 05:31:06,549 - util.base_logger - INFO - Epoch=5 done. (autoencoder.py:252)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 05:32:11,938 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 05:32:11,938 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 05:32:11,938 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48 already exists (utils.py:25)[0m
[33;20m2022-04-28 05:32:11,938 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 05:32:11,938 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 05:32:24,965 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 05:32:24,965 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 05:32:24,965 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48 already exists (utils.py:25)[0m
[33;20m2022-04-28 05:32:24,965 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 05:32:24,965 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 05:55:52,627 - util.base_logger - INFO - Epoch=6 done. (autoencoder.py:252)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 05:56:52,399 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 05:56:52,400 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 05:56:52,400 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48 already exists (utils.py:25)[0m
[33;20m2022-04-28 05:56:52,400 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 05:56:52,400 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 05:57:06,455 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 05:57:06,455 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 05:57:06,455 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48 already exists (utils.py:25)[0m
[33;20m2022-04-28 05:57:06,455 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 05:57:06,455 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 06:20:33,440 - util.base_logger - INFO - Epoch=7 done. (autoencoder.py:252)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 06:21:32,400 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 06:21:32,401 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 06:21:32,401 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48 already exists (utils.py:25)[0m
[33;20m2022-04-28 06:21:32,401 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 06:21:32,401 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 06:21:45,040 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 06:21:45,040 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 06:21:45,040 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48 already exists (utils.py:25)[0m
[33;20m2022-04-28 06:21:45,041 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 06:21:45,041 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 06:45:14,351 - util.base_logger - INFO - Epoch=8 done. (autoencoder.py:252)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 06:46:17,644 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 06:46:17,645 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 06:46:17,645 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48 already exists (utils.py:25)[0m
[33;20m2022-04-28 06:46:17,645 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 06:46:17,645 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 06:46:33,397 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 06:46:33,398 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 06:46:33,398 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48 already exists (utils.py:25)[0m
[33;20m2022-04-28 06:46:33,398 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 06:46:33,398 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 07:09:57,940 - util.base_logger - INFO - Epoch=9 done. (autoencoder.py:252)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 07:11:10,349 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 07:11:10,349 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 07:11:10,349 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48 already exists (utils.py:25)[0m
[33;20m2022-04-28 07:11:10,350 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 07:11:10,352 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 07:11:22,167 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 07:11:22,167 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 07:11:22,167 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48 already exists (utils.py:25)[0m
[33;20m2022-04-28 07:11:22,167 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 07:11:22,167 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 07:34:57,301 - util.base_logger - INFO - Epoch=10 done. (autoencoder.py:252)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 07:35:55,276 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 07:35:55,276 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 07:35:55,276 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48 already exists (utils.py:25)[0m
[33;20m2022-04-28 07:35:55,276 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 07:35:55,278 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 07:36:12,065 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 07:36:12,066 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 07:36:12,066 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48 already exists (utils.py:25)[0m
[33;20m2022-04-28 07:36:12,066 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 07:36:12,066 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 07:59:39,000 - util.base_logger - INFO - Epoch=11 done. (autoencoder.py:252)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 08:00:40,919 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 08:00:40,919 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 08:00:40,920 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48 already exists (utils.py:25)[0m
[33;20m2022-04-28 08:00:40,920 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 08:00:40,920 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 08:00:56,115 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 08:00:56,115 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 08:00:56,116 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48 already exists (utils.py:25)[0m
[33;20m2022-04-28 08:00:56,116 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 08:00:56,116 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 08:24:21,890 - util.base_logger - INFO - Epoch=12 done. (autoencoder.py:252)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 08:25:32,113 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 08:25:32,113 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 08:25:32,113 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48 already exists (utils.py:25)[0m
[33;20m2022-04-28 08:25:32,113 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 08:25:32,113 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 08:25:46,262 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 08:25:46,262 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 08:25:46,263 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48 already exists (utils.py:25)[0m
[33;20m2022-04-28 08:25:46,263 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 08:25:46,263 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 08:49:21,238 - util.base_logger - INFO - Epoch=13 done. (autoencoder.py:252)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 08:50:20,637 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 08:50:20,637 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 08:50:20,637 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48 already exists (utils.py:25)[0m
[33;20m2022-04-28 08:50:20,637 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 08:50:20,637 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 08:50:33,562 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 08:50:33,562 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 08:50:33,562 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48 already exists (utils.py:25)[0m
[33;20m2022-04-28 08:50:33,562 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 08:50:33,562 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 09:14:01,529 - util.base_logger - INFO - Epoch=14 done. (autoencoder.py:252)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 09:15:04,529 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 09:15:04,529 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 09:15:04,529 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48 already exists (utils.py:25)[0m
[33;20m2022-04-28 09:15:04,530 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 09:15:04,530 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 09:15:18,291 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 09:15:18,291 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 09:15:18,291 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48 already exists (utils.py:25)[0m
[33;20m2022-04-28 09:15:18,291 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 09:15:18,291 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 09:38:28,176 - util.base_logger - INFO - Epoch=15 done. (autoencoder.py:252)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 09:39:26,428 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 09:39:26,429 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 09:39:26,429 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48 already exists (utils.py:25)[0m
[33;20m2022-04-28 09:39:26,429 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 09:39:26,430 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 09:39:42,849 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 09:39:42,849 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 09:39:42,849 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48 already exists (utils.py:25)[0m
[33;20m2022-04-28 09:39:42,849 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 09:39:42,849 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 10:02:32,534 - util.base_logger - INFO - Epoch=16 done. (autoencoder.py:252)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 10:03:37,406 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 10:03:37,406 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 10:03:37,406 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48 already exists (utils.py:25)[0m
[33;20m2022-04-28 10:03:37,406 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 10:03:37,423 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 10:03:50,759 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 10:03:50,759 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 10:03:50,759 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48 already exists (utils.py:25)[0m
[33;20m2022-04-28 10:03:50,759 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 10:03:50,759 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 10:27:00,162 - util.base_logger - INFO - Epoch=17 done. (autoencoder.py:252)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 10:28:15,896 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 10:28:15,897 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 10:28:15,897 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48 already exists (utils.py:25)[0m
[33;20m2022-04-28 10:28:15,897 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 10:28:15,913 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 10:28:32,383 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 10:28:32,383 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 10:28:32,383 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48 already exists (utils.py:25)[0m
[33;20m2022-04-28 10:28:32,383 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 10:28:32,383 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 10:52:13,169 - util.base_logger - INFO - Epoch=18 done. (autoencoder.py:252)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 10:53:24,282 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 10:53:24,282 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 10:53:24,282 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48 already exists (utils.py:25)[0m
[33;20m2022-04-28 10:53:24,284 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 10:53:24,285 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 10:53:37,362 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 10:53:37,362 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 10:53:37,362 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48 already exists (utils.py:25)[0m
[33;20m2022-04-28 10:53:37,362 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 10:53:37,362 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 11:16:41,099 - util.base_logger - INFO - Epoch=19 done. (autoencoder.py:252)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 11:17:42,366 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 11:17:42,366 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 11:17:42,367 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48 already exists (utils.py:25)[0m
[33;20m2022-04-28 11:17:42,367 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 11:17:42,367 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 11:17:55,368 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 11:17:55,368 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 11:17:55,368 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48 already exists (utils.py:25)[0m
[33;20m2022-04-28 11:17:55,368 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 11:17:55,368 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 11:41:03,177 - util.base_logger - INFO - Epoch=20 done. (autoencoder.py:252)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 11:42:36,685 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 11:42:36,685 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 11:42:36,685 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48 already exists (utils.py:25)[0m
[33;20m2022-04-28 11:42:36,685 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 11:42:36,686 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 11:42:49,363 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 11:42:49,363 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 11:42:49,363 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48 already exists (utils.py:25)[0m
[33;20m2022-04-28 11:42:49,363 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 11:42:49,363 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 12:06:28,544 - util.base_logger - INFO - Epoch=21 done. (autoencoder.py:252)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 12:07:28,021 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 12:07:28,021 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 12:07:28,021 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48 already exists (utils.py:25)[0m
[33;20m2022-04-28 12:07:28,021 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 12:07:28,022 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 12:07:44,535 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 12:07:44,535 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 12:07:44,535 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48 already exists (utils.py:25)[0m
[33;20m2022-04-28 12:07:44,535 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 12:07:44,535 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 12:31:26,485 - util.base_logger - INFO - Epoch=22 done. (autoencoder.py:252)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 12:32:40,119 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 12:32:40,119 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 12:32:40,119 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48 already exists (utils.py:25)[0m
[33;20m2022-04-28 12:32:40,120 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 12:32:40,121 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 12:32:53,513 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 12:32:53,513 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 12:32:53,513 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48 already exists (utils.py:25)[0m
[33;20m2022-04-28 12:32:53,514 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 12:32:53,514 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 12:56:28,438 - util.base_logger - INFO - Epoch=23 done. (autoencoder.py:252)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 12:57:21,833 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 12:57:21,833 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 12:57:21,833 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48 already exists (utils.py:25)[0m
[33;20m2022-04-28 12:57:21,834 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 12:57:21,835 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 12:57:33,422 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 12:57:33,422 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 12:57:33,422 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48 already exists (utils.py:25)[0m
[33;20m2022-04-28 12:57:33,423 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 12:57:33,423 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 13:21:07,349 - util.base_logger - INFO - Epoch=24 done. (autoencoder.py:252)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 13:22:02,181 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 13:22:02,182 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 13:22:02,182 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48 already exists (utils.py:25)[0m
[33;20m2022-04-28 13:22:02,182 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 13:22:02,183 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 13:22:15,474 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 13:22:15,474 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 13:22:15,474 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48 already exists (utils.py:25)[0m
[33;20m2022-04-28 13:22:15,474 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 13:22:15,475 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-02-37-48/CovAE/gray-scale already exists (utils.py:25)[0m
