[33;20m2022-04-28 02:37:47,465 - util.base_logger - WARNING - ./out already exists (utils.py:25)[0m
[32m2022-04-28 02:37:47,469 - util.base_logger - INFO - Created folder ./out/cluster_50-28-04-2022-02-37-47 (utils.py:28)[0m
[32m2022-04-28 02:37:47,471 - util.base_logger - INFO - Program starting in cluster-mode (main.py:64)[0m
[32m2022-04-28 02:37:47,471 - util.base_logger - INFO - STARTING PROGRAM (main.py:73)[0m
[32m2022-04-28 02:37:47,471 - util.base_logger - INFO - Selected parameters: (main.py:74)[0m
[32m2022-04-28 02:37:47,471 - util.base_logger - INFO - {'name': 'cluster_50-28-04-2022-02-37-47', 'train': True, 'model': './models/models-autoencodergray-scale.pth', 'letters': ['alpha', 'beta', 'chi', 'delta', 'epsilon', 'eta', 'gamma', 'iota', 'kappa', 'lambda', 'mu', 'nu', 'omega', 'omicron', 'phi', 'pi', 'psi', 'rho', 'sigma', 'tau', 'theta', 'xi', 'ypsilon', 'zeta'], 'logging': 40, 'mode': 'cluster', 'epochs': 50, 'dimensions': 28, 'tqdm': False, 'processing': 'gray-scale', 'root': 'out/cluster_50-28-04-2022-02-37-47'} (main.py:75)[0m
[32m2022-04-28 02:37:47,664 - util.base_logger - INFO - Network(
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
[32m2022-04-28 02:37:47,667 - util.base_logger - INFO - ConvAutoEncoder(
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
[32m2022-04-28 02:38:38,705 - util.base_logger - INFO - Epoch=0 done. (autoencoder.py:252)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 02:39:37,423 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 02:39:37,431 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 02:39:37,431 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47 already exists (utils.py:25)[0m
[32m2022-04-28 02:39:37,432 - util.base_logger - INFO - Created folder ././out/cluster_50-28-04-2022-02-37-47/CovAE (utils.py:28)[0m
[32m2022-04-28 02:39:37,433 - util.base_logger - INFO - Created folder ././out/cluster_50-28-04-2022-02-37-47/CovAE/gray-scale (utils.py:28)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 02:39:58,839 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 02:39:58,839 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 02:39:58,839 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47 already exists (utils.py:25)[0m
[33;20m2022-04-28 02:39:58,839 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 02:39:58,839 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 02:59:59,477 - util.base_logger - INFO - Epoch=1 done. (autoencoder.py:252)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 03:00:56,339 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 03:00:56,345 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 03:00:56,345 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47 already exists (utils.py:25)[0m
[33;20m2022-04-28 03:00:56,345 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 03:00:56,345 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 03:01:07,949 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 03:01:07,949 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 03:01:07,950 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47 already exists (utils.py:25)[0m
[33;20m2022-04-28 03:01:07,950 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 03:01:07,950 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 03:21:11,793 - util.base_logger - INFO - Epoch=2 done. (autoencoder.py:252)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 03:22:08,294 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 03:22:08,294 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 03:22:08,294 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47 already exists (utils.py:25)[0m
[33;20m2022-04-28 03:22:08,294 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 03:22:08,294 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 03:22:18,236 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 03:22:18,237 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 03:22:18,237 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47 already exists (utils.py:25)[0m
[33;20m2022-04-28 03:22:18,237 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 03:22:18,237 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 03:42:18,519 - util.base_logger - INFO - Epoch=3 done. (autoencoder.py:252)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 03:43:15,953 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 03:43:15,954 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 03:43:15,954 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47 already exists (utils.py:25)[0m
[33;20m2022-04-28 03:43:15,954 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 03:43:15,954 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 03:43:25,806 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 03:43:25,806 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 03:43:25,806 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47 already exists (utils.py:25)[0m
[33;20m2022-04-28 03:43:25,806 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 03:43:25,806 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 04:03:31,014 - util.base_logger - INFO - Epoch=4 done. (autoencoder.py:252)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 04:04:36,560 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 04:04:36,560 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 04:04:36,560 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47 already exists (utils.py:25)[0m
[33;20m2022-04-28 04:04:36,560 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 04:04:36,560 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 04:04:47,571 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 04:04:47,571 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 04:04:47,571 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47 already exists (utils.py:25)[0m
[33;20m2022-04-28 04:04:47,571 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 04:04:47,571 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 04:24:52,396 - util.base_logger - INFO - Epoch=5 done. (autoencoder.py:252)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 04:25:54,433 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 04:25:54,434 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 04:25:54,434 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47 already exists (utils.py:25)[0m
[33;20m2022-04-28 04:25:54,434 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 04:25:54,434 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 04:26:05,223 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 04:26:05,223 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 04:26:05,223 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47 already exists (utils.py:25)[0m
[33;20m2022-04-28 04:26:05,223 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 04:26:05,223 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 04:46:14,762 - util.base_logger - INFO - Epoch=6 done. (autoencoder.py:252)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 04:47:15,175 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 04:47:15,175 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 04:47:15,175 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47 already exists (utils.py:25)[0m
[33;20m2022-04-28 04:47:15,175 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 04:47:15,175 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 04:47:26,143 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 04:47:26,144 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 04:47:26,144 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47 already exists (utils.py:25)[0m
[33;20m2022-04-28 04:47:26,144 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 04:47:26,144 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 05:07:34,560 - util.base_logger - INFO - Epoch=7 done. (autoencoder.py:252)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 05:08:34,765 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 05:08:34,765 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 05:08:34,765 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47 already exists (utils.py:25)[0m
[33;20m2022-04-28 05:08:34,765 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 05:08:34,766 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 05:08:47,275 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 05:08:47,276 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 05:08:47,276 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47 already exists (utils.py:25)[0m
[33;20m2022-04-28 05:08:47,276 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 05:08:47,276 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 05:28:54,823 - util.base_logger - INFO - Epoch=8 done. (autoencoder.py:252)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 05:29:53,008 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 05:29:53,009 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 05:29:53,009 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47 already exists (utils.py:25)[0m
[33;20m2022-04-28 05:29:53,009 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 05:29:53,009 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 05:30:03,017 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 05:30:03,017 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 05:30:03,017 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47 already exists (utils.py:25)[0m
[33;20m2022-04-28 05:30:03,017 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 05:30:03,017 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 05:49:56,169 - util.base_logger - INFO - Epoch=9 done. (autoencoder.py:252)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 05:50:49,744 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 05:50:49,744 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 05:50:49,744 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47 already exists (utils.py:25)[0m
[33;20m2022-04-28 05:50:49,744 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 05:50:49,744 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 05:50:59,560 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 05:50:59,560 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 05:50:59,560 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47 already exists (utils.py:25)[0m
[33;20m2022-04-28 05:50:59,560 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 05:50:59,561 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 06:11:12,390 - util.base_logger - INFO - Epoch=10 done. (autoencoder.py:252)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 06:12:10,389 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 06:12:10,390 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 06:12:10,390 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47 already exists (utils.py:25)[0m
[33;20m2022-04-28 06:12:10,390 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 06:12:10,390 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 06:12:23,043 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 06:12:23,043 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 06:12:23,044 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47 already exists (utils.py:25)[0m
[33;20m2022-04-28 06:12:23,044 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 06:12:23,044 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 06:32:35,067 - util.base_logger - INFO - Epoch=11 done. (autoencoder.py:252)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 06:33:30,879 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 06:33:30,880 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 06:33:30,880 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47 already exists (utils.py:25)[0m
[33;20m2022-04-28 06:33:30,880 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 06:33:30,880 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 06:33:42,500 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 06:33:42,501 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 06:33:42,501 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47 already exists (utils.py:25)[0m
[33;20m2022-04-28 06:33:42,501 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 06:33:42,501 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 06:53:43,884 - util.base_logger - INFO - Epoch=12 done. (autoencoder.py:252)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 06:54:41,508 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 06:54:41,508 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 06:54:41,508 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47 already exists (utils.py:25)[0m
[33;20m2022-04-28 06:54:41,508 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 06:54:41,508 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 06:54:52,518 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 06:54:52,518 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 06:54:52,518 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47 already exists (utils.py:25)[0m
[33;20m2022-04-28 06:54:52,518 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 06:54:52,518 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 07:15:00,364 - util.base_logger - INFO - Epoch=13 done. (autoencoder.py:252)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 07:16:04,084 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 07:16:04,084 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 07:16:04,084 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47 already exists (utils.py:25)[0m
[33;20m2022-04-28 07:16:04,084 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 07:16:04,084 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 07:16:16,599 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 07:16:16,599 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 07:16:16,599 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47 already exists (utils.py:25)[0m
[33;20m2022-04-28 07:16:16,600 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 07:16:16,600 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 07:36:26,667 - util.base_logger - INFO - Epoch=14 done. (autoencoder.py:252)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 07:37:24,090 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 07:37:24,090 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 07:37:24,090 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47 already exists (utils.py:25)[0m
[33;20m2022-04-28 07:37:24,090 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 07:37:24,090 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 07:37:35,274 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 07:37:35,274 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 07:37:35,274 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47 already exists (utils.py:25)[0m
[33;20m2022-04-28 07:37:35,274 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 07:37:35,274 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 07:57:46,593 - util.base_logger - INFO - Epoch=15 done. (autoencoder.py:252)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 07:58:47,710 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 07:58:47,710 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 07:58:47,710 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47 already exists (utils.py:25)[0m
[33;20m2022-04-28 07:58:47,710 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 07:58:47,710 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 07:58:59,622 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 07:58:59,622 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 07:58:59,622 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47 already exists (utils.py:25)[0m
[33;20m2022-04-28 07:58:59,622 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 07:58:59,622 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 08:19:15,468 - util.base_logger - INFO - Epoch=16 done. (autoencoder.py:252)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 08:20:10,850 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 08:20:10,850 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 08:20:10,850 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47 already exists (utils.py:25)[0m
[33;20m2022-04-28 08:20:10,850 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 08:20:10,850 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 08:20:20,708 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 08:20:20,708 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 08:20:20,708 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47 already exists (utils.py:25)[0m
[33;20m2022-04-28 08:20:20,708 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 08:20:20,708 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 08:40:46,808 - util.base_logger - INFO - Epoch=17 done. (autoencoder.py:252)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 08:41:45,665 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 08:41:45,665 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 08:41:45,665 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47 already exists (utils.py:25)[0m
[33;20m2022-04-28 08:41:45,666 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 08:41:45,666 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 08:41:55,485 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 08:41:55,485 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 08:41:55,485 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47 already exists (utils.py:25)[0m
[33;20m2022-04-28 08:41:55,485 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 08:41:55,485 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 09:02:03,391 - util.base_logger - INFO - Epoch=18 done. (autoencoder.py:252)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 09:03:05,196 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 09:03:05,196 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 09:03:05,196 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47 already exists (utils.py:25)[0m
[33;20m2022-04-28 09:03:05,196 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 09:03:05,197 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 09:03:15,220 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 09:03:15,220 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 09:03:15,220 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47 already exists (utils.py:25)[0m
[33;20m2022-04-28 09:03:15,220 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 09:03:15,220 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 09:23:20,041 - util.base_logger - INFO - Epoch=19 done. (autoencoder.py:252)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 09:24:21,861 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 09:24:21,861 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 09:24:21,861 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47 already exists (utils.py:25)[0m
[33;20m2022-04-28 09:24:21,862 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 09:24:21,863 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 09:24:34,156 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 09:24:34,156 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 09:24:34,156 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47 already exists (utils.py:25)[0m
[33;20m2022-04-28 09:24:34,157 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 09:24:34,157 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 09:44:50,001 - util.base_logger - INFO - Epoch=20 done. (autoencoder.py:252)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 09:45:52,861 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 09:45:52,862 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 09:45:52,862 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47 already exists (utils.py:25)[0m
[33;20m2022-04-28 09:45:52,863 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 09:45:52,864 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 09:46:05,105 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 09:46:05,105 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 09:46:05,105 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47 already exists (utils.py:25)[0m
[33;20m2022-04-28 09:46:05,105 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 09:46:05,105 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 10:06:31,631 - util.base_logger - INFO - Epoch=21 done. (autoencoder.py:252)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 10:07:37,133 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 10:07:37,133 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 10:07:37,133 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47 already exists (utils.py:25)[0m
[33;20m2022-04-28 10:07:37,133 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 10:07:37,133 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 10:07:47,830 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 10:07:47,830 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 10:07:47,830 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47 already exists (utils.py:25)[0m
[33;20m2022-04-28 10:07:47,830 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 10:07:47,830 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 10:28:14,019 - util.base_logger - INFO - Epoch=22 done. (autoencoder.py:252)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 10:29:14,229 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 10:29:14,229 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 10:29:14,229 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47 already exists (utils.py:25)[0m
[33;20m2022-04-28 10:29:14,229 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 10:29:14,229 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 10:29:24,789 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 10:29:24,789 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 10:29:24,789 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47 already exists (utils.py:25)[0m
[33;20m2022-04-28 10:29:24,789 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 10:29:24,789 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 10:49:47,069 - util.base_logger - INFO - Epoch=23 done. (autoencoder.py:252)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 10:50:51,135 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 10:50:51,136 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 10:50:51,136 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47 already exists (utils.py:25)[0m
[33;20m2022-04-28 10:50:51,137 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 10:50:51,138 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 10:51:03,840 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 10:51:03,841 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 10:51:03,841 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47 already exists (utils.py:25)[0m
[33;20m2022-04-28 10:51:03,841 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 10:51:03,841 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 11:11:20,611 - util.base_logger - INFO - Epoch=24 done. (autoencoder.py:252)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 11:12:23,588 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 11:12:23,588 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 11:12:23,588 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47 already exists (utils.py:25)[0m
[33;20m2022-04-28 11:12:23,590 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 11:12:23,590 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 11:12:33,981 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 11:12:33,982 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 11:12:33,982 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47 already exists (utils.py:25)[0m
[33;20m2022-04-28 11:12:33,982 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 11:12:33,983 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 11:32:47,981 - util.base_logger - INFO - Epoch=25 done. (autoencoder.py:252)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 11:33:50,938 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 11:33:50,938 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 11:33:50,938 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47 already exists (utils.py:25)[0m
[33;20m2022-04-28 11:33:50,938 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 11:33:50,938 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 11:34:00,781 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 11:34:00,781 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 11:34:00,781 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47 already exists (utils.py:25)[0m
[33;20m2022-04-28 11:34:00,781 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 11:34:00,781 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 11:54:11,218 - util.base_logger - INFO - Epoch=26 done. (autoencoder.py:252)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 11:55:14,206 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 11:55:14,206 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 11:55:14,206 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47 already exists (utils.py:25)[0m
[33;20m2022-04-28 11:55:14,207 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 11:55:14,207 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 11:55:24,349 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 11:55:24,349 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 11:55:24,349 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47 already exists (utils.py:25)[0m
[33;20m2022-04-28 11:55:24,349 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 11:55:24,350 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 12:15:42,413 - util.base_logger - INFO - Epoch=27 done. (autoencoder.py:252)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 12:16:41,629 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 12:16:41,629 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 12:16:41,629 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47 already exists (utils.py:25)[0m
[33;20m2022-04-28 12:16:41,629 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 12:16:41,629 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 12:16:52,136 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 12:16:52,136 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 12:16:52,136 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47 already exists (utils.py:25)[0m
[33;20m2022-04-28 12:16:52,136 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 12:16:52,137 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 12:37:09,487 - util.base_logger - INFO - Epoch=28 done. (autoencoder.py:252)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 12:38:06,186 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 12:38:06,197 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 12:38:06,197 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47 already exists (utils.py:25)[0m
[33;20m2022-04-28 12:38:06,199 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 12:38:06,200 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 12:38:16,418 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 12:38:16,418 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 12:38:16,418 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47 already exists (utils.py:25)[0m
[33;20m2022-04-28 12:38:16,418 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 12:38:16,418 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-02-37-47/CovAE/gray-scale already exists (utils.py:25)[0m
