[33;20m2022-04-28 14:47:02,156 - util.base_logger - WARNING - ./out already exists (utils.py:25)[0m
[32m2022-04-28 14:47:02,178 - util.base_logger - INFO - Created folder ./out/cluster_100-28-04-2022-14-47-02 (utils.py:28)[0m
[32m2022-04-28 14:47:02,179 - util.base_logger - INFO - Program starting in cluster-mode (main.py:64)[0m
[32m2022-04-28 14:47:02,179 - util.base_logger - INFO - STARTING PROGRAM (main.py:73)[0m
[32m2022-04-28 14:47:02,179 - util.base_logger - INFO - Selected parameters: (main.py:74)[0m
[32m2022-04-28 14:47:02,179 - util.base_logger - INFO - {'name': 'cluster_100', 'name_time': 'cluster_100-28-04-2022-14-47-02', 'train': True, 'model': './models/models-autoencodergray-scale.pth', 'letters': ['alpha', 'beta', 'chi', 'delta', 'epsilon', 'eta', 'gamma', 'iota', 'kappa', 'lambda', 'mu', 'nu', 'omega', 'omicron', 'phi', 'pi', 'psi', 'rho', 'sigma', 'tau', 'theta', 'xi', 'ypsilon', 'zeta'], 'logging': 40, 'mode': 'cluster', 'epochs': 100, 'dimensions': 28, 'tqdm': False, 'processing': 'gray-scale', 'root': 'out/cluster_100-28-04-2022-14-47-02'} (main.py:75)[0m
[32m2022-04-28 14:47:02,398 - util.base_logger - INFO - Network(
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
) (autoencoder.py:195)[0m
[32m2022-04-28 14:47:02,402 - util.base_logger - INFO - ConvAutoEncoder(
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
) (autoencoder.py:201)[0m
[32m2022-04-28 14:48:03,946 - util.base_logger - INFO - Epoch=0 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 14:48:05,479 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 14:48:05,485 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 14:48:05,485 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[32m2022-04-28 14:48:05,486 - util.base_logger - INFO - Created folder ././out/cluster_100-28-04-2022-14-47-02/CovAE (utils.py:28)[0m
[32m2022-04-28 14:48:05,486 - util.base_logger - INFO - Created folder ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale (utils.py:28)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 14:48:15,569 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 14:48:15,570 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 14:48:15,570 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 14:48:15,570 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 14:48:15,570 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 14:49:10,467 - util.base_logger - INFO - Epoch=1 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 14:49:11,821 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 14:49:11,827 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 14:49:11,827 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 14:49:11,827 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 14:49:11,827 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 14:49:15,501 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 14:49:15,501 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 14:49:15,502 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 14:49:15,502 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 14:49:15,502 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 14:50:11,698 - util.base_logger - INFO - Epoch=2 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 14:50:13,151 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 14:50:13,151 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 14:50:13,151 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 14:50:13,151 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 14:50:13,151 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 14:50:16,826 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 14:50:16,827 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 14:50:16,827 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 14:50:16,827 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 14:50:16,827 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 14:51:13,828 - util.base_logger - INFO - Epoch=3 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 14:51:15,327 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 14:51:15,327 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 14:51:15,327 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 14:51:15,327 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 14:51:15,327 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 14:51:18,993 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 14:51:18,993 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 14:51:18,993 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 14:51:18,993 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 14:51:18,993 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 14:52:14,064 - util.base_logger - INFO - Epoch=4 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 14:52:15,552 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 14:52:15,552 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 14:52:15,552 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 14:52:15,552 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 14:52:15,552 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 14:52:19,227 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 14:52:19,227 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 14:52:19,228 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 14:52:19,228 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 14:52:19,228 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 14:53:14,397 - util.base_logger - INFO - Epoch=5 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 14:53:15,916 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 14:53:15,916 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 14:53:15,916 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 14:53:15,916 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 14:53:15,916 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 14:53:19,639 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 14:53:19,639 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 14:53:19,639 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 14:53:19,639 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 14:53:19,640 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 14:54:14,852 - util.base_logger - INFO - Epoch=6 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 14:54:16,338 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 14:54:16,339 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 14:54:16,339 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 14:54:16,339 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 14:54:16,339 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 14:54:20,024 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 14:54:20,024 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 14:54:20,024 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 14:54:20,024 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 14:54:20,025 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 14:55:15,234 - util.base_logger - INFO - Epoch=7 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 14:55:16,598 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 14:55:16,598 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 14:55:16,598 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 14:55:16,598 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 14:55:16,598 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 14:55:20,317 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 14:55:20,317 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 14:55:20,317 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 14:55:20,317 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 14:55:20,317 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 14:56:15,554 - util.base_logger - INFO - Epoch=8 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 14:56:17,072 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 14:56:17,073 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 14:56:17,073 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 14:56:17,073 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 14:56:17,073 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 14:56:20,768 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 14:56:20,769 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 14:56:20,769 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 14:56:20,769 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 14:56:20,769 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 14:57:16,002 - util.base_logger - INFO - Epoch=9 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 14:57:17,503 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 14:57:17,503 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 14:57:17,503 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 14:57:17,503 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 14:57:17,503 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 14:57:21,189 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 14:57:21,189 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 14:57:21,189 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 14:57:21,189 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 14:57:21,189 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 14:58:16,285 - util.base_logger - INFO - Epoch=10 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 14:58:17,792 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 14:58:17,793 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 14:58:17,793 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 14:58:17,793 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 14:58:17,793 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 14:58:21,484 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 14:58:21,484 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 14:58:21,485 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 14:58:21,485 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 14:58:21,485 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 14:59:16,735 - util.base_logger - INFO - Epoch=11 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 14:59:18,252 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 14:59:18,252 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 14:59:18,252 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 14:59:18,253 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 14:59:18,253 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 14:59:21,943 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 14:59:21,943 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 14:59:21,943 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 14:59:21,943 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 14:59:21,943 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 15:00:17,171 - util.base_logger - INFO - Epoch=12 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:00:18,661 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:00:18,682 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:00:18,682 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:00:18,682 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:00:18,682 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:00:22,372 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:00:22,373 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:00:22,373 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:00:22,373 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:00:22,373 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 15:01:17,664 - util.base_logger - INFO - Epoch=13 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:01:19,147 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:01:19,147 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:01:19,147 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:01:19,147 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:01:19,147 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:01:22,865 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:01:22,865 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:01:22,865 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:01:22,865 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:01:22,865 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 15:02:18,078 - util.base_logger - INFO - Epoch=14 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:02:19,529 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:02:19,530 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:02:19,530 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:02:19,530 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:02:19,530 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:02:23,254 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:02:23,254 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:02:23,254 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:02:23,254 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:02:23,255 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 15:03:18,552 - util.base_logger - INFO - Epoch=15 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:03:20,061 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:03:20,062 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:03:20,062 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:03:20,062 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:03:20,062 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:03:23,782 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:03:23,783 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:03:23,783 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:03:23,783 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:03:23,783 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 15:04:19,119 - util.base_logger - INFO - Epoch=16 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:04:20,487 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:04:20,488 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:04:20,488 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:04:20,488 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:04:20,488 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:04:24,205 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:04:24,206 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:04:24,206 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:04:24,206 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:04:24,206 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 15:05:19,610 - util.base_logger - INFO - Epoch=17 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:05:21,102 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:05:21,102 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:05:21,102 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:05:21,102 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:05:21,102 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:05:24,834 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:05:24,835 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:05:24,835 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:05:24,835 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:05:24,835 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 15:06:20,213 - util.base_logger - INFO - Epoch=18 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:06:21,708 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:06:21,709 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:06:21,709 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:06:21,709 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:06:21,709 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:06:25,211 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:06:25,212 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:06:25,212 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:06:25,212 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:06:25,212 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 15:07:20,827 - util.base_logger - INFO - Epoch=19 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:07:22,304 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:07:22,306 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:07:22,306 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:07:22,306 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:07:22,306 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:07:25,838 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:07:25,838 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:07:25,838 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:07:25,838 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:07:25,838 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 15:08:21,439 - util.base_logger - INFO - Epoch=20 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:08:22,891 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:08:22,892 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:08:22,892 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:08:22,892 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:08:22,892 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:08:26,419 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:08:26,419 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:08:26,419 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:08:26,419 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:08:26,419 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 15:09:21,950 - util.base_logger - INFO - Epoch=21 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:09:23,384 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:09:23,384 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:09:23,384 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:09:23,384 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:09:23,384 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:09:26,871 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:09:26,871 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:09:26,871 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:09:26,871 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:09:26,871 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 15:10:22,388 - util.base_logger - INFO - Epoch=22 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:10:23,874 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:10:23,874 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:10:23,875 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:10:23,875 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:10:23,875 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:10:27,371 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:10:27,371 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:10:27,371 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:10:27,371 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:10:27,371 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 15:11:22,884 - util.base_logger - INFO - Epoch=23 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:11:24,377 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:11:24,377 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:11:24,377 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:11:24,377 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:11:24,377 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:11:27,891 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:11:27,891 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:11:27,891 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:11:27,891 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:11:27,891 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 15:12:23,446 - util.base_logger - INFO - Epoch=24 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:12:24,933 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:12:24,933 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:12:24,933 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:12:24,934 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:12:24,934 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:12:28,448 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:12:28,448 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:12:28,448 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:12:28,448 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:12:28,448 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 15:13:23,853 - util.base_logger - INFO - Epoch=25 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:13:25,569 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:13:25,569 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:13:25,570 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:13:25,570 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:13:25,570 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:13:29,078 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:13:29,078 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:13:29,078 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:13:29,078 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:13:29,078 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 15:14:24,452 - util.base_logger - INFO - Epoch=26 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:14:26,055 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:14:26,055 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:14:26,055 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:14:26,055 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:14:26,056 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:14:29,587 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:14:29,588 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:14:29,588 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:14:29,588 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:14:29,588 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 15:15:25,023 - util.base_logger - INFO - Epoch=27 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:15:26,739 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:15:26,739 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:15:26,739 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:15:26,739 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:15:26,739 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:15:30,251 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:15:30,251 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:15:30,251 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:15:30,251 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:15:30,252 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 15:16:25,595 - util.base_logger - INFO - Epoch=28 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:16:27,097 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:16:27,097 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:16:27,097 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:16:27,097 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:16:27,097 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:16:30,827 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:16:30,828 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:16:30,828 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:16:30,828 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:16:30,828 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 15:17:26,229 - util.base_logger - INFO - Epoch=29 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:17:27,729 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:17:27,730 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:17:27,730 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:17:27,730 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:17:27,730 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:17:31,457 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:17:31,457 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:17:31,457 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:17:31,457 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:17:31,457 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 15:18:26,918 - util.base_logger - INFO - Epoch=30 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:18:28,431 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:18:28,432 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:18:28,432 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:18:28,432 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:18:28,432 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:18:32,174 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:18:32,175 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:18:32,175 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:18:32,175 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:18:32,175 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 15:19:27,557 - util.base_logger - INFO - Epoch=31 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:19:29,050 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:19:29,050 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:19:29,050 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:19:29,050 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:19:29,050 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:19:32,785 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:19:32,786 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:19:32,786 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:19:32,786 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:19:32,786 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 15:20:28,276 - util.base_logger - INFO - Epoch=32 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:20:29,778 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:20:29,779 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:20:29,779 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:20:29,779 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:20:29,779 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:20:33,506 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:20:33,506 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:20:33,506 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:20:33,506 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:20:33,507 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 15:21:28,988 - util.base_logger - INFO - Epoch=33 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:21:30,508 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:21:30,508 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:21:30,508 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:21:30,508 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:21:30,508 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:21:34,248 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:21:34,248 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:21:34,248 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:21:34,248 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:21:34,248 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 15:22:29,795 - util.base_logger - INFO - Epoch=34 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:22:31,303 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:22:31,303 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:22:31,303 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:22:31,303 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:22:31,303 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:22:34,799 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:22:34,799 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:22:34,799 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:22:34,799 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:22:34,799 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 15:23:30,323 - util.base_logger - INFO - Epoch=35 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:23:32,076 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:23:32,076 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:23:32,076 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:23:32,076 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:23:32,076 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:23:35,600 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:23:35,600 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:23:35,600 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:23:35,601 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:23:35,601 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 15:24:31,054 - util.base_logger - INFO - Epoch=36 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:24:32,780 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:24:32,780 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:24:32,780 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:24:32,780 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:24:32,780 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:24:36,301 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:24:36,301 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:24:36,302 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:24:36,302 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:24:36,302 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 15:25:31,723 - util.base_logger - INFO - Epoch=37 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:25:33,136 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:25:33,136 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:25:33,136 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:25:33,136 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:25:33,136 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:25:36,925 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:25:36,925 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:25:36,925 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:25:36,925 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:25:36,926 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 15:26:32,331 - util.base_logger - INFO - Epoch=38 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:26:33,827 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:26:33,827 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:26:33,828 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:26:33,828 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:26:33,828 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:26:37,567 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:26:37,567 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:26:37,567 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:26:37,567 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:26:37,567 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 15:27:32,997 - util.base_logger - INFO - Epoch=39 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:27:34,489 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:27:34,489 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:27:34,489 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:27:34,489 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:27:34,489 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:27:38,228 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:27:38,228 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:27:38,228 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:27:38,228 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:27:38,229 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 15:28:33,571 - util.base_logger - INFO - Epoch=40 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:28:35,064 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:28:35,064 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:28:35,064 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:28:35,065 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:28:35,065 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:28:38,824 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:28:38,824 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:28:38,824 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:28:38,824 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:28:38,824 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 15:29:34,188 - util.base_logger - INFO - Epoch=41 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:29:35,575 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:29:35,576 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:29:35,576 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:29:35,576 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:29:35,576 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:29:39,091 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:29:39,091 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:29:39,092 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:29:39,092 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:29:39,092 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 15:30:34,431 - util.base_logger - INFO - Epoch=42 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:30:36,090 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:30:36,091 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:30:36,091 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:30:36,091 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:30:36,091 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:30:39,609 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:30:39,609 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:30:39,609 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:30:39,609 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:30:39,609 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 15:31:34,949 - util.base_logger - INFO - Epoch=43 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:31:36,449 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:31:36,449 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:31:36,449 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:31:36,449 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:31:36,449 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:31:40,240 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:31:40,240 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:31:40,240 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:31:40,252 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:31:40,252 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 15:32:35,666 - util.base_logger - INFO - Epoch=44 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:32:37,174 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:32:37,174 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:32:37,174 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:32:37,174 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:32:37,174 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:32:40,929 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:32:40,929 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:32:40,930 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:32:40,930 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:32:40,930 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 15:33:36,292 - util.base_logger - INFO - Epoch=45 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:33:37,792 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:33:37,792 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:33:37,792 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:33:37,793 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:33:37,793 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:33:41,566 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:33:41,566 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:33:41,566 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:33:41,567 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:33:41,567 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 15:34:36,926 - util.base_logger - INFO - Epoch=46 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:34:38,440 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:34:38,440 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:34:38,441 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:34:38,441 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:34:38,441 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:34:41,939 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:34:41,940 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:34:41,940 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:34:41,940 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:34:41,940 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 15:35:37,592 - util.base_logger - INFO - Epoch=47 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:35:39,099 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:35:39,099 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:35:39,099 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:35:39,099 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:35:39,099 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:35:42,638 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:35:42,639 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:35:42,639 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:35:42,639 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:35:42,639 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 15:36:37,980 - util.base_logger - INFO - Epoch=48 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:36:39,680 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:36:39,680 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:36:39,681 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:36:39,681 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:36:39,681 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:36:43,179 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:36:43,179 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:36:43,179 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:36:43,179 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:36:43,179 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 15:37:38,415 - util.base_logger - INFO - Epoch=49 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:37:39,913 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:37:39,913 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:37:39,913 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:37:39,913 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:37:39,913 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:37:43,685 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:37:43,685 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:37:43,685 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:37:43,686 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:37:43,686 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 15:38:38,797 - util.base_logger - INFO - Epoch=50 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:38:40,292 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:38:40,292 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:38:40,292 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:38:40,292 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:38:40,292 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:38:44,050 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:38:44,050 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:38:44,050 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:38:44,051 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:38:44,051 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 15:39:39,175 - util.base_logger - INFO - Epoch=51 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:39:40,655 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:39:40,655 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:39:40,655 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:39:40,655 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:39:40,655 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:39:44,419 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:39:44,420 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:39:44,420 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:39:44,420 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:39:44,420 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 15:40:39,551 - util.base_logger - INFO - Epoch=52 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:40:41,032 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:40:41,033 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:40:41,033 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:40:41,033 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:40:41,033 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:40:44,547 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:40:44,547 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:40:44,547 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:40:44,547 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:40:44,547 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 15:41:39,790 - util.base_logger - INFO - Epoch=53 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:41:41,516 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:41:41,516 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:41:41,516 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:41:41,516 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:41:41,516 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:41:45,028 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:41:45,029 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:41:45,029 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:41:45,029 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:41:45,029 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 15:42:40,236 - util.base_logger - INFO - Epoch=54 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:42:41,736 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:42:41,736 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:42:41,736 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:42:41,736 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:42:41,737 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:42:45,518 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:42:45,518 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:42:45,518 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:42:45,518 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:42:45,519 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 15:43:40,734 - util.base_logger - INFO - Epoch=55 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:43:42,148 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:43:42,148 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:43:42,149 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:43:42,149 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:43:42,149 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:43:45,903 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:43:45,903 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:43:45,904 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:43:45,904 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:43:45,904 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 15:44:41,069 - util.base_logger - INFO - Epoch=56 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:44:42,554 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:44:42,554 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:44:42,554 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:44:42,555 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:44:42,555 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:44:46,323 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:44:46,323 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:44:46,323 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:44:46,323 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:44:46,324 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 15:45:41,457 - util.base_logger - INFO - Epoch=57 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:45:42,945 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:45:42,945 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:45:42,945 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:45:42,946 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:45:42,946 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:45:46,465 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:45:46,465 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:45:46,465 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:45:46,466 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:45:46,466 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 15:46:41,748 - util.base_logger - INFO - Epoch=58 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:46:43,509 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:46:43,509 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:46:43,510 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:46:43,510 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:46:43,510 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:46:47,038 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:46:47,038 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:46:47,039 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:46:47,039 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:46:47,039 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 15:47:42,322 - util.base_logger - INFO - Epoch=59 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:47:43,810 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:47:43,810 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:47:43,811 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:47:43,811 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:47:43,811 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:47:47,638 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:47:47,654 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:47:47,654 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:47:47,654 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:47:47,654 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 15:48:42,900 - util.base_logger - INFO - Epoch=60 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:48:44,383 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:48:44,383 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:48:44,383 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:48:44,383 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:48:44,383 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:48:48,161 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:48:48,161 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:48:48,161 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:48:48,161 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:48:48,161 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 15:49:43,364 - util.base_logger - INFO - Epoch=61 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:49:44,888 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:49:44,889 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:49:44,889 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:49:44,889 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:49:44,889 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:49:48,672 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:49:48,673 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:49:48,673 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:49:48,673 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:49:48,673 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 15:50:43,822 - util.base_logger - INFO - Epoch=62 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:50:45,309 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:50:45,310 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:50:45,310 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:50:45,310 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:50:45,310 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:50:48,832 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:50:48,832 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:50:48,833 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:50:48,833 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:50:48,833 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 15:51:44,015 - util.base_logger - INFO - Epoch=63 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:51:45,798 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:51:45,799 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:51:45,799 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:51:45,799 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:51:45,799 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:51:49,345 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:51:49,346 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:51:49,346 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:51:49,346 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:51:49,346 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 15:52:44,655 - util.base_logger - INFO - Epoch=64 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:52:46,094 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:52:46,094 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:52:46,095 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:52:46,095 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:52:46,095 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:52:49,896 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:52:49,896 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:52:49,896 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:52:49,896 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:52:49,896 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 15:53:45,231 - util.base_logger - INFO - Epoch=65 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:53:46,707 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:53:46,707 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:53:46,707 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:53:46,707 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:53:46,707 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:53:50,529 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:53:50,529 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:53:50,529 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:53:50,529 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:53:50,529 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 15:54:45,758 - util.base_logger - INFO - Epoch=66 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:54:47,196 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:54:47,196 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:54:47,196 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:54:47,196 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:54:47,196 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:54:50,730 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:54:50,730 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:54:50,730 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:54:50,730 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:54:50,730 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 15:55:45,970 - util.base_logger - INFO - Epoch=67 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:55:47,658 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:55:47,658 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:55:47,658 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:55:47,658 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:55:47,658 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:55:51,173 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:55:51,173 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:55:51,173 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:55:51,174 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:55:51,174 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 15:56:46,215 - util.base_logger - INFO - Epoch=68 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:56:47,671 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:56:47,672 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:56:47,672 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:56:47,672 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:56:47,672 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:56:51,501 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:56:51,501 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:56:51,501 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:56:51,502 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:56:51,502 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 15:57:46,678 - util.base_logger - INFO - Epoch=69 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:57:48,145 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:57:48,146 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:57:48,146 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:57:48,146 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:57:48,146 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:57:51,929 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:57:51,930 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:57:51,930 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:57:51,930 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:57:51,930 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 15:58:47,031 - util.base_logger - INFO - Epoch=70 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:58:48,458 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:58:48,458 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:58:48,458 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:58:48,458 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:58:48,458 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:58:51,986 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:58:51,986 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:58:51,986 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:58:51,986 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:58:51,987 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 15:59:47,124 - util.base_logger - INFO - Epoch=71 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:59:48,818 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:59:48,818 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:59:48,818 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:59:48,819 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:59:48,819 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 15:59:52,382 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 15:59:52,382 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 15:59:52,382 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 15:59:52,382 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 15:59:52,382 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 16:00:47,431 - util.base_logger - INFO - Epoch=72 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:00:48,793 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:00:48,794 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:00:48,794 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:00:48,794 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:00:48,794 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:00:52,620 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:00:52,620 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:00:52,621 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:00:52,621 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:00:52,621 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 16:01:47,352 - util.base_logger - INFO - Epoch=73 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:01:48,777 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:01:48,777 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:01:48,777 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:01:48,777 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:01:48,777 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:01:52,567 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:01:52,567 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:01:52,567 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:01:52,567 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:01:52,567 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 16:02:47,236 - util.base_logger - INFO - Epoch=74 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:02:48,757 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:02:48,757 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:02:48,757 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:02:48,757 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:02:48,757 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:02:52,235 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:02:52,235 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:02:52,235 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:02:52,235 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:02:52,235 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 16:03:47,050 - util.base_logger - INFO - Epoch=75 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:03:48,853 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:03:48,853 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:03:48,853 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:03:48,854 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:03:48,854 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:03:52,362 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:03:52,362 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:03:52,362 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:03:52,362 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:03:52,362 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 16:04:47,151 - util.base_logger - INFO - Epoch=76 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:04:48,597 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:04:48,597 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:04:48,597 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:04:48,597 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:04:48,597 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:04:52,364 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:04:52,364 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:04:52,364 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:04:52,364 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:04:52,364 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 16:05:47,141 - util.base_logger - INFO - Epoch=77 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:05:48,988 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:05:48,988 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:05:48,989 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:05:48,989 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:05:48,989 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:05:52,483 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:05:52,484 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:05:52,484 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:05:52,484 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:05:52,484 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 16:06:47,592 - util.base_logger - INFO - Epoch=78 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:06:49,083 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:06:49,083 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:06:49,083 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:06:49,083 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:06:49,083 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:06:52,591 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:06:52,591 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:06:52,591 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:06:52,591 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:06:52,591 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 16:07:47,604 - util.base_logger - INFO - Epoch=79 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:07:49,105 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:07:49,105 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:07:49,105 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:07:49,106 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:07:49,106 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:07:52,908 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:07:52,908 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:07:52,908 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:07:52,908 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:07:52,908 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 16:08:47,911 - util.base_logger - INFO - Epoch=80 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:08:49,425 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:08:49,426 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:08:49,426 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:08:49,426 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:08:49,426 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:08:53,177 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:08:53,177 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:08:53,177 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:08:53,177 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:08:53,177 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 16:09:48,153 - util.base_logger - INFO - Epoch=81 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:09:49,513 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:09:49,513 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:09:49,514 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:09:49,514 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:09:49,514 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:09:53,003 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:09:53,004 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:09:53,004 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:09:53,004 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:09:53,004 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 16:10:48,114 - util.base_logger - INFO - Epoch=82 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:10:49,919 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:10:49,919 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:10:49,919 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:10:49,919 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:10:49,919 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:10:53,411 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:10:53,411 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:10:53,411 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:10:53,411 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:10:53,411 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 16:11:48,512 - util.base_logger - INFO - Epoch=83 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:11:50,024 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:11:50,025 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:11:50,025 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:11:50,025 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:11:50,025 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:11:53,826 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:11:53,826 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:11:53,826 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:11:53,826 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:11:53,826 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 16:12:48,914 - util.base_logger - INFO - Epoch=84 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:12:50,417 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:12:50,418 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:12:50,418 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:12:50,418 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:12:50,418 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:12:53,901 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:12:53,901 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:12:53,901 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:12:53,901 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:12:53,901 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 16:13:49,332 - util.base_logger - INFO - Epoch=85 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:13:50,767 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:13:50,767 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:13:50,767 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:13:50,768 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:13:50,768 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:13:54,281 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:13:54,282 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:13:54,282 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:13:54,282 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:13:54,282 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 16:14:49,298 - util.base_logger - INFO - Epoch=86 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:14:50,684 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:14:50,684 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:14:50,684 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:14:50,684 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:14:50,684 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:14:54,500 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:14:54,501 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:14:54,501 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:14:54,501 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:14:54,501 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 16:15:49,577 - util.base_logger - INFO - Epoch=87 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:15:51,068 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:15:51,068 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:15:51,068 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:15:51,069 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:15:51,069 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:15:54,896 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:15:54,896 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:15:54,896 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:15:54,897 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:15:54,897 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 16:16:49,933 - util.base_logger - INFO - Epoch=88 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:16:51,446 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:16:51,447 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:16:51,447 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:16:51,447 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:16:51,447 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:16:54,947 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:16:54,947 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:16:54,947 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:16:54,947 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:16:54,947 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 16:17:49,798 - util.base_logger - INFO - Epoch=89 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:17:51,590 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:17:51,590 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:17:51,591 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:17:51,591 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:17:51,591 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:17:55,080 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:17:55,080 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:17:55,080 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:17:55,080 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:17:55,080 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 16:18:49,885 - util.base_logger - INFO - Epoch=90 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:18:51,256 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:18:51,256 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:18:51,256 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:18:51,256 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:18:51,256 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:18:55,021 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:18:55,021 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:18:55,021 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:18:55,021 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:18:55,021 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 16:19:49,829 - util.base_logger - INFO - Epoch=91 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:19:51,324 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:19:51,325 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:19:51,325 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:19:51,326 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:19:51,326 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:19:54,823 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:19:54,823 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:19:54,824 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:19:54,824 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:19:54,824 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 16:20:49,773 - util.base_logger - INFO - Epoch=92 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:20:51,471 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:20:51,472 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:20:51,472 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:20:51,472 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:20:51,472 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:20:54,981 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:20:54,981 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:20:54,981 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:20:54,981 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:20:54,981 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 16:21:49,941 - util.base_logger - INFO - Epoch=93 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:21:51,315 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:21:51,316 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:21:51,316 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:21:51,316 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:21:51,316 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:21:55,104 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:21:55,104 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:21:55,104 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:21:55,104 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:21:55,104 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 16:22:49,970 - util.base_logger - INFO - Epoch=94 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:22:51,475 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:22:51,475 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:22:51,475 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:22:51,475 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:22:51,475 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:22:54,988 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:22:54,988 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:22:54,988 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:22:54,988 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:22:54,988 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 16:23:49,870 - util.base_logger - INFO - Epoch=95 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:23:51,661 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:23:51,662 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:23:51,662 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:23:51,662 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:23:51,663 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:23:55,187 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:23:55,187 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:23:55,187 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:23:55,188 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:23:55,188 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 16:24:50,216 - util.base_logger - INFO - Epoch=96 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:24:51,615 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:24:51,616 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:24:51,616 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:24:51,616 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:24:51,616 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:24:55,431 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:24:55,431 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:24:55,431 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:24:55,431 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:24:55,431 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 16:25:50,380 - util.base_logger - INFO - Epoch=97 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:25:51,882 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:25:51,883 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:25:51,883 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:25:51,883 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:25:51,883 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:25:55,418 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:25:55,418 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:25:55,418 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:25:55,418 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:25:55,418 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 16:26:50,437 - util.base_logger - INFO - Epoch=98 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:26:52,161 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:26:52,161 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:26:52,161 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:26:52,161 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:26:52,161 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:26:55,666 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:26:55,667 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:26:55,667 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:26:55,667 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:26:55,667 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 16:27:50,665 - util.base_logger - INFO - Epoch=99 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:27:52,139 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:27:52,140 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:27:52,140 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:27:52,140 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:27:52,140 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:27:55,958 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:27:55,958 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:27:55,958 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:27:55,958 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:27:55,959 - util.base_logger - WARNING - ././out/cluster_100-28-04-2022-14-47-02/CovAE/gray-scale already exists (utils.py:25)[0m
