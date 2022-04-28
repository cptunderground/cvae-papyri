[33;20m2022-04-28 02:38:20,799 - util.base_logger - WARNING - ./out already exists (utils.py:25)[0m
[32m2022-04-28 02:38:20,805 - util.base_logger - INFO - Created folder ./out/cluster_30-28-04-2022-02-38-20 (utils.py:28)[0m
[32m2022-04-28 02:38:20,806 - util.base_logger - INFO - Program starting in cluster-mode (main.py:64)[0m
[32m2022-04-28 02:38:20,806 - util.base_logger - INFO - STARTING PROGRAM (main.py:73)[0m
[32m2022-04-28 02:38:20,806 - util.base_logger - INFO - Selected parameters: (main.py:74)[0m
[32m2022-04-28 02:38:20,806 - util.base_logger - INFO - {'name': 'cluster_30-28-04-2022-02-38-20', 'train': True, 'model': './models/models-autoencodergray-scale.pth', 'letters': ['alpha', 'beta', 'chi', 'delta', 'epsilon', 'eta', 'gamma', 'iota', 'kappa', 'lambda', 'mu', 'nu', 'omega', 'omicron', 'phi', 'pi', 'psi', 'rho', 'sigma', 'tau', 'theta', 'xi', 'ypsilon', 'zeta'], 'logging': 40, 'mode': 'cluster', 'epochs': 30, 'dimensions': 28, 'tqdm': False, 'processing': 'gray-scale', 'root': 'out/cluster_30-28-04-2022-02-38-20'} (main.py:75)[0m
[32m2022-04-28 02:38:21,020 - util.base_logger - INFO - Network(
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
[32m2022-04-28 02:38:21,023 - util.base_logger - INFO - ConvAutoEncoder(
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
[32m2022-04-28 02:39:39,004 - util.base_logger - INFO - Epoch=0 done. (autoencoder.py:252)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 02:40:40,090 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 02:40:40,101 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 02:40:40,101 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20 already exists (utils.py:25)[0m
[32m2022-04-28 02:40:40,102 - util.base_logger - INFO - Created folder ././out/cluster_30-28-04-2022-02-38-20/CovAE (utils.py:28)[0m
[32m2022-04-28 02:40:40,102 - util.base_logger - INFO - Created folder ././out/cluster_30-28-04-2022-02-38-20/CovAE/gray-scale (utils.py:28)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 02:41:06,755 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 02:41:06,755 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 02:41:06,755 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20 already exists (utils.py:25)[0m
[33;20m2022-04-28 02:41:06,755 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 02:41:06,755 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 03:01:54,278 - util.base_logger - INFO - Epoch=1 done. (autoencoder.py:252)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 03:02:57,620 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 03:02:57,629 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 03:02:57,629 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20 already exists (utils.py:25)[0m
[33;20m2022-04-28 03:02:57,629 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 03:02:57,629 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 03:03:08,964 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 03:03:08,964 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 03:03:08,964 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20 already exists (utils.py:25)[0m
[33;20m2022-04-28 03:03:08,964 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 03:03:08,965 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 03:23:56,685 - util.base_logger - INFO - Epoch=2 done. (autoencoder.py:252)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 03:24:59,384 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 03:24:59,384 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 03:24:59,384 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20 already exists (utils.py:25)[0m
[33;20m2022-04-28 03:24:59,384 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 03:24:59,385 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 03:25:10,676 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 03:25:10,676 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 03:25:10,676 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20 already exists (utils.py:25)[0m
[33;20m2022-04-28 03:25:10,676 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 03:25:10,676 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 03:45:53,991 - util.base_logger - INFO - Epoch=3 done. (autoencoder.py:252)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 03:46:56,169 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 03:46:56,169 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 03:46:56,169 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20 already exists (utils.py:25)[0m
[33;20m2022-04-28 03:46:56,170 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 03:46:56,170 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 03:47:07,740 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 03:47:07,741 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 03:47:07,741 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20 already exists (utils.py:25)[0m
[33;20m2022-04-28 03:47:07,741 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 03:47:07,741 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 04:07:46,963 - util.base_logger - INFO - Epoch=4 done. (autoencoder.py:252)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 04:08:48,309 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 04:08:48,309 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 04:08:48,309 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20 already exists (utils.py:25)[0m
[33;20m2022-04-28 04:08:48,310 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 04:08:48,310 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 04:08:59,081 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 04:08:59,081 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 04:08:59,081 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20 already exists (utils.py:25)[0m
[33;20m2022-04-28 04:08:59,082 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 04:08:59,082 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 04:29:39,291 - util.base_logger - INFO - Epoch=5 done. (autoencoder.py:252)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 04:30:40,285 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 04:30:40,285 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 04:30:40,285 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20 already exists (utils.py:25)[0m
[33;20m2022-04-28 04:30:40,285 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 04:30:40,285 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 04:30:50,643 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 04:30:50,643 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 04:30:50,643 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20 already exists (utils.py:25)[0m
[33;20m2022-04-28 04:30:50,643 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 04:30:50,643 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 04:51:29,472 - util.base_logger - INFO - Epoch=6 done. (autoencoder.py:252)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 04:52:25,374 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 04:52:25,374 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 04:52:25,374 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20 already exists (utils.py:25)[0m
[33;20m2022-04-28 04:52:25,374 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 04:52:25,376 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 04:52:35,443 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 04:52:35,443 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 04:52:35,443 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20 already exists (utils.py:25)[0m
[33;20m2022-04-28 04:52:35,443 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 04:52:35,443 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 05:13:05,943 - util.base_logger - INFO - Epoch=7 done. (autoencoder.py:252)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 05:14:00,962 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 05:14:00,962 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 05:14:00,963 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20 already exists (utils.py:25)[0m
[33;20m2022-04-28 05:14:00,963 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 05:14:00,963 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 05:14:10,918 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 05:14:10,918 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 05:14:10,918 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20 already exists (utils.py:25)[0m
[33;20m2022-04-28 05:14:10,918 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 05:14:10,918 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 05:34:35,805 - util.base_logger - INFO - Epoch=8 done. (autoencoder.py:252)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 05:35:31,918 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 05:35:31,918 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 05:35:31,918 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20 already exists (utils.py:25)[0m
[33;20m2022-04-28 05:35:31,918 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 05:35:31,918 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 05:35:42,099 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 05:35:42,099 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 05:35:42,099 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20 already exists (utils.py:25)[0m
[33;20m2022-04-28 05:35:42,099 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 05:35:42,099 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 05:56:12,853 - util.base_logger - INFO - Epoch=9 done. (autoencoder.py:252)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 05:57:08,814 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 05:57:08,815 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 05:57:08,815 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20 already exists (utils.py:25)[0m
[33;20m2022-04-28 05:57:08,815 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 05:57:08,816 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 05:57:21,186 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 05:57:21,187 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 05:57:21,187 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20 already exists (utils.py:25)[0m
[33;20m2022-04-28 05:57:21,187 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 05:57:21,187 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 06:18:07,773 - util.base_logger - INFO - Epoch=10 done. (autoencoder.py:252)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 06:19:03,217 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 06:19:03,218 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 06:19:03,218 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20 already exists (utils.py:25)[0m
[33;20m2022-04-28 06:19:03,218 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 06:19:03,218 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 06:19:13,475 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 06:19:13,475 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 06:19:13,475 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20 already exists (utils.py:25)[0m
[33;20m2022-04-28 06:19:13,475 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 06:19:13,475 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 06:39:51,329 - util.base_logger - INFO - Epoch=11 done. (autoencoder.py:252)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 06:40:49,605 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 06:40:49,605 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 06:40:49,605 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20 already exists (utils.py:25)[0m
[33;20m2022-04-28 06:40:49,605 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 06:40:49,607 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 06:41:01,559 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 06:41:01,559 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 06:41:01,559 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20 already exists (utils.py:25)[0m
[33;20m2022-04-28 06:41:01,559 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 06:41:01,559 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 07:01:48,118 - util.base_logger - INFO - Epoch=12 done. (autoencoder.py:252)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 07:02:48,714 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 07:02:48,715 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 07:02:48,715 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20 already exists (utils.py:25)[0m
[33;20m2022-04-28 07:02:48,715 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 07:02:48,715 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 07:02:59,055 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 07:02:59,055 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 07:02:59,055 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20 already exists (utils.py:25)[0m
[33;20m2022-04-28 07:02:59,055 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 07:02:59,055 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 07:23:41,532 - util.base_logger - INFO - Epoch=13 done. (autoencoder.py:252)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 07:24:45,406 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 07:24:45,407 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 07:24:45,407 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20 already exists (utils.py:25)[0m
[33;20m2022-04-28 07:24:45,407 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 07:24:45,407 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 07:24:56,626 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 07:24:56,626 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 07:24:56,626 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20 already exists (utils.py:25)[0m
[33;20m2022-04-28 07:24:56,626 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 07:24:56,626 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 07:45:40,140 - util.base_logger - INFO - Epoch=14 done. (autoencoder.py:252)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 07:46:42,141 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 07:46:42,141 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 07:46:42,141 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20 already exists (utils.py:25)[0m
[33;20m2022-04-28 07:46:42,141 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 07:46:42,141 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 07:46:55,233 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 07:46:55,233 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 07:46:55,233 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20 already exists (utils.py:25)[0m
[33;20m2022-04-28 07:46:55,233 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 07:46:55,233 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 08:07:37,667 - util.base_logger - INFO - Epoch=15 done. (autoencoder.py:252)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 08:08:40,507 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 08:08:40,507 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 08:08:40,507 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20 already exists (utils.py:25)[0m
[33;20m2022-04-28 08:08:40,508 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 08:08:40,508 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 08:08:51,668 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 08:08:51,668 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 08:08:51,668 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20 already exists (utils.py:25)[0m
[33;20m2022-04-28 08:08:51,669 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 08:08:51,669 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 08:29:35,334 - util.base_logger - INFO - Epoch=16 done. (autoencoder.py:252)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 08:30:42,175 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 08:30:42,176 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 08:30:42,176 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20 already exists (utils.py:25)[0m
[33;20m2022-04-28 08:30:42,176 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 08:30:42,176 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 08:30:53,479 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 08:30:53,480 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 08:30:53,480 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20 already exists (utils.py:25)[0m
[33;20m2022-04-28 08:30:53,480 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 08:30:53,480 - util.base_logger - WARNING - ././out/cluster_30-28-04-2022-02-38-20/CovAE/gray-scale already exists (utils.py:25)[0m
slurmstepd: error: *** JOB 46051649 ON shi130 CANCELLED AT 2022-04-28T08:38:03 DUE TO TIME LIMIT ***
