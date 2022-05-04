[33;20m2022-04-28 16:28:13,854 - util.base_logger - WARNING - ./out already exists (utils.py:25)[0m
[32m2022-04-28 16:28:13,855 - util.base_logger - INFO - Created folder ./out/cluster_50-28-04-2022-16-28-13 (utils.py:28)[0m
[32m2022-04-28 16:28:13,855 - util.base_logger - INFO - Program starting in cluster-mode (main.py:64)[0m
[32m2022-04-28 16:28:13,855 - util.base_logger - INFO - STARTING PROGRAM (main.py:73)[0m
[32m2022-04-28 16:28:13,855 - util.base_logger - INFO - Selected parameters: (main.py:74)[0m
[32m2022-04-28 16:28:13,855 - util.base_logger - INFO - {'name': 'cluster_50', 'name_time': 'cluster_50-28-04-2022-16-28-13', 'train': True, 'model': './models/models-autoencodergray-scale.pth', 'letters': ['alpha', 'beta', 'chi', 'delta', 'epsilon', 'eta', 'gamma', 'iota', 'kappa', 'lambda', 'mu', 'nu', 'omega', 'omicron', 'phi', 'pi', 'psi', 'rho', 'sigma', 'tau', 'theta', 'xi', 'ypsilon', 'zeta'], 'logging': 40, 'mode': 'cluster', 'epochs': 50, 'dimensions': 28, 'tqdm': False, 'processing': 'gray-scale', 'root': 'out/cluster_50-28-04-2022-16-28-13'} (main.py:75)[0m
[32m2022-04-28 16:28:14,079 - util.base_logger - INFO - Network(
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
[32m2022-04-28 16:28:14,082 - util.base_logger - INFO - ConvAutoEncoder(
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
[32m2022-04-28 16:29:13,964 - util.base_logger - INFO - Epoch=0 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:29:15,385 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:29:15,392 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:29:15,392 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[32m2022-04-28 16:29:15,392 - util.base_logger - INFO - Created folder ././out/cluster_50-28-04-2022-16-28-13/CovAE (utils.py:28)[0m
[32m2022-04-28 16:29:15,393 - util.base_logger - INFO - Created folder ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale (utils.py:28)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:29:25,443 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:29:25,443 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:29:25,443 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:29:25,443 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:29:25,443 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 16:30:21,985 - util.base_logger - INFO - Epoch=1 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:30:23,489 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:30:23,493 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:30:23,493 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:30:23,493 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:30:23,493 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:30:27,203 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:30:27,203 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:30:27,203 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:30:27,203 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:30:27,203 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 16:31:23,872 - util.base_logger - INFO - Epoch=2 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:31:25,336 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:31:25,336 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:31:25,336 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:31:25,337 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:31:25,337 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:31:29,031 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:31:29,031 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:31:29,031 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:31:29,031 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:31:29,031 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 16:32:25,661 - util.base_logger - INFO - Epoch=3 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:32:27,203 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:32:27,203 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:32:27,203 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:32:27,203 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:32:27,204 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:32:30,911 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:32:30,911 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:32:30,911 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:32:30,911 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:32:30,911 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 16:33:27,541 - util.base_logger - INFO - Epoch=4 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:33:28,955 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:33:28,956 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:33:28,956 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:33:28,956 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:33:28,956 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:33:32,668 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:33:32,668 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:33:32,668 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:33:32,668 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:33:32,668 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 16:34:29,277 - util.base_logger - INFO - Epoch=5 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:34:30,769 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:34:30,769 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:34:30,769 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:34:30,769 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:34:30,769 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:34:34,499 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:34:34,499 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:34:34,499 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:34:34,499 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:34:34,500 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 16:35:31,130 - util.base_logger - INFO - Epoch=6 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:35:32,570 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:35:32,570 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:35:32,570 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:35:32,570 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:35:32,570 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:35:36,290 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:35:36,290 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:35:36,290 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:35:36,291 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:35:36,291 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 16:36:32,880 - util.base_logger - INFO - Epoch=7 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:36:34,290 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:36:34,290 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:36:34,290 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:36:34,290 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:36:34,290 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:36:38,033 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:36:38,034 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:36:38,034 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:36:38,034 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:36:38,034 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 16:37:34,753 - util.base_logger - INFO - Epoch=8 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:37:36,291 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:37:36,291 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:37:36,291 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:37:36,291 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:37:36,291 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:37:39,979 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:37:39,979 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:37:39,979 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:37:39,979 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:37:39,979 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 16:38:36,672 - util.base_logger - INFO - Epoch=9 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:38:38,214 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:38:38,214 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:38:38,214 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:38:38,215 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:38:38,215 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:38:41,923 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:38:41,924 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:38:41,924 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:38:41,924 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:38:41,924 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 16:39:38,562 - util.base_logger - INFO - Epoch=10 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:39:40,080 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:39:40,081 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:39:40,081 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:39:40,081 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:39:40,081 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:39:43,782 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:39:43,782 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:39:43,782 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:39:43,782 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:39:43,782 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 16:40:40,592 - util.base_logger - INFO - Epoch=11 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:40:42,100 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:40:42,100 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:40:42,100 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:40:42,100 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:40:42,101 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:40:45,830 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:40:45,830 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:40:45,831 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:40:45,831 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:40:45,831 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 16:41:42,607 - util.base_logger - INFO - Epoch=12 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:41:44,112 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:41:44,113 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:41:44,113 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:41:44,113 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:41:44,113 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:41:47,862 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:41:47,862 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:41:47,862 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:41:47,863 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:41:47,863 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 16:42:44,816 - util.base_logger - INFO - Epoch=13 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:42:46,325 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:42:46,325 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:42:46,325 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:42:46,326 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:42:46,326 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:42:50,063 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:42:50,064 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:42:50,064 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:42:50,064 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:42:50,064 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 16:43:46,947 - util.base_logger - INFO - Epoch=14 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:43:48,341 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:43:48,341 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:43:48,342 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:43:48,342 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:43:48,342 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:43:52,072 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:43:52,072 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:43:52,072 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:43:52,072 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:43:52,072 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 16:44:48,793 - util.base_logger - INFO - Epoch=15 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:44:50,325 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:44:50,326 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:44:50,326 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:44:50,326 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:44:50,326 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:44:54,070 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:44:54,071 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:44:54,071 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:44:54,071 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:44:54,071 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 16:45:50,921 - util.base_logger - INFO - Epoch=16 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:45:52,441 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:45:52,441 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:45:52,441 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:45:52,441 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:45:52,441 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:45:56,178 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:45:56,179 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:45:56,179 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:45:56,179 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:45:56,179 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 16:46:53,029 - util.base_logger - INFO - Epoch=17 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:46:54,548 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:46:54,548 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:46:54,548 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:46:54,548 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:46:54,549 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:46:58,324 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:46:58,325 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:46:58,325 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:46:58,325 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:46:58,325 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 16:47:55,097 - util.base_logger - INFO - Epoch=18 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:47:56,612 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:47:56,612 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:47:56,612 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:47:56,612 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:47:56,613 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:48:00,153 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:48:00,154 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:48:00,154 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:48:00,154 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:48:00,154 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 16:48:57,247 - util.base_logger - INFO - Epoch=19 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:48:58,672 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:48:58,672 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:48:58,672 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:48:58,672 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:48:58,672 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:49:02,217 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:49:02,217 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:49:02,217 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:49:02,217 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:49:02,217 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 16:49:59,519 - util.base_logger - INFO - Epoch=20 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:50:01,038 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:50:01,039 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:50:01,039 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:50:01,039 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:50:01,039 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:50:04,590 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:50:04,590 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:50:04,590 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:50:04,590 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:50:04,590 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 16:51:01,788 - util.base_logger - INFO - Epoch=21 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:51:03,325 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:51:03,326 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:51:03,326 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:51:03,326 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:51:03,326 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:51:06,876 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:51:06,876 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:51:06,876 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:51:06,876 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:51:06,876 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 16:52:04,072 - util.base_logger - INFO - Epoch=22 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:52:05,605 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:52:05,605 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:52:05,605 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:52:05,605 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:52:05,606 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:52:09,129 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:52:09,129 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:52:09,129 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:52:09,130 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:52:09,130 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 16:53:06,264 - util.base_logger - INFO - Epoch=23 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:53:07,670 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:53:07,670 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:53:07,670 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:53:07,670 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:53:07,670 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:53:11,215 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:53:11,215 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:53:11,215 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:53:11,215 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:53:11,215 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 16:54:08,320 - util.base_logger - INFO - Epoch=24 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:54:09,821 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:54:09,821 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:54:09,821 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:54:09,821 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:54:09,822 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:54:13,370 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:54:13,371 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:54:13,371 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:54:13,371 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:54:13,371 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 16:55:10,521 - util.base_logger - INFO - Epoch=25 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:55:12,065 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:55:12,066 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:55:12,066 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:55:12,066 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:55:12,066 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:55:15,585 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:55:15,585 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:55:15,586 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:55:15,586 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:55:15,586 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 16:56:12,465 - util.base_logger - INFO - Epoch=26 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:56:14,030 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:56:14,030 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:56:14,031 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:56:14,031 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:56:14,031 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:56:17,544 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:56:17,544 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:56:17,544 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:56:17,544 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:56:17,544 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 16:57:14,344 - util.base_logger - INFO - Epoch=27 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:57:16,093 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:57:16,093 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:57:16,093 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:57:16,093 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:57:16,094 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:57:19,611 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:57:19,611 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:57:19,611 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:57:19,611 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:57:19,611 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 16:58:16,621 - util.base_logger - INFO - Epoch=28 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:58:18,268 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:58:18,268 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:58:18,268 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:58:18,268 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:58:18,268 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:58:21,815 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:58:21,815 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:58:21,815 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:58:21,815 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:58:21,815 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 16:59:18,725 - util.base_logger - INFO - Epoch=29 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:59:20,248 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:59:20,249 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:59:20,249 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:59:20,249 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:59:20,249 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 16:59:24,024 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 16:59:24,024 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 16:59:24,024 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 16:59:24,024 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 16:59:24,024 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 17:00:20,944 - util.base_logger - INFO - Epoch=30 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 17:00:22,480 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 17:00:22,480 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 17:00:22,480 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 17:00:22,480 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 17:00:22,480 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 17:00:26,251 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 17:00:26,251 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 17:00:26,252 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 17:00:26,252 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 17:00:26,252 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 17:01:23,115 - util.base_logger - INFO - Epoch=31 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 17:01:24,566 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 17:01:24,566 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 17:01:24,567 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 17:01:24,567 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 17:01:24,567 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 17:01:28,325 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 17:01:28,326 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 17:01:28,326 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 17:01:28,326 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 17:01:28,326 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 17:02:25,213 - util.base_logger - INFO - Epoch=32 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 17:02:26,717 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 17:02:26,717 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 17:02:26,717 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 17:02:26,717 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 17:02:26,717 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 17:02:30,465 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 17:02:30,465 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 17:02:30,465 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 17:02:30,465 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 17:02:30,465 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 17:03:27,280 - util.base_logger - INFO - Epoch=33 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 17:03:28,672 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 17:03:28,672 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 17:03:28,672 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 17:03:28,672 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 17:03:28,672 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 17:03:32,447 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 17:03:32,447 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 17:03:32,447 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 17:03:32,447 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 17:03:32,447 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 17:04:29,279 - util.base_logger - INFO - Epoch=34 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 17:04:30,677 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 17:04:30,678 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 17:04:30,678 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 17:04:30,678 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 17:04:30,678 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 17:04:34,227 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 17:04:34,228 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 17:04:34,228 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 17:04:34,228 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 17:04:34,228 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 17:05:31,392 - util.base_logger - INFO - Epoch=35 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 17:05:32,912 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 17:05:32,912 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 17:05:32,912 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 17:05:32,912 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 17:05:32,912 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 17:05:36,443 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 17:05:36,444 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 17:05:36,444 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 17:05:36,444 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 17:05:36,444 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 17:06:33,349 - util.base_logger - INFO - Epoch=36 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 17:06:35,106 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 17:06:35,106 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 17:06:35,106 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 17:06:35,106 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 17:06:35,106 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 17:06:38,644 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 17:06:38,645 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 17:06:38,645 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 17:06:38,645 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 17:06:38,645 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 17:07:35,463 - util.base_logger - INFO - Epoch=37 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 17:07:36,985 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 17:07:36,985 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 17:07:36,985 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 17:07:36,985 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 17:07:36,985 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 17:07:40,798 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 17:07:40,799 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 17:07:40,799 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 17:07:40,799 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 17:07:40,799 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 17:08:37,636 - util.base_logger - INFO - Epoch=38 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 17:08:39,168 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 17:08:39,168 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 17:08:39,168 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 17:08:39,168 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 17:08:39,169 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 17:08:42,927 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 17:08:42,927 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 17:08:42,928 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 17:08:42,928 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 17:08:42,928 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 17:09:39,807 - util.base_logger - INFO - Epoch=39 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 17:09:41,341 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 17:09:41,341 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 17:09:41,341 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 17:09:41,341 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 17:09:41,341 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 17:09:45,135 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 17:09:45,136 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 17:09:45,136 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 17:09:45,136 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 17:09:45,136 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 17:10:42,257 - util.base_logger - INFO - Epoch=40 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 17:10:43,711 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 17:10:43,712 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 17:10:43,712 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 17:10:43,712 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 17:10:43,712 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 17:10:47,536 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 17:10:47,537 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 17:10:47,537 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 17:10:47,537 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 17:10:47,537 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 17:11:44,873 - util.base_logger - INFO - Epoch=41 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 17:11:46,397 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 17:11:46,397 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 17:11:46,397 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 17:11:46,397 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 17:11:46,398 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 17:11:49,977 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 17:11:49,977 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 17:11:49,977 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 17:11:49,977 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 17:11:49,977 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 17:12:47,428 - util.base_logger - INFO - Epoch=42 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 17:12:48,885 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 17:12:48,885 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 17:12:48,885 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 17:12:48,885 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 17:12:48,885 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 17:12:52,446 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 17:12:52,446 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 17:12:52,447 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 17:12:52,447 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 17:12:52,447 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 17:13:49,769 - util.base_logger - INFO - Epoch=43 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 17:13:51,412 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 17:13:51,412 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 17:13:51,412 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 17:13:51,412 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 17:13:51,412 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 17:13:54,982 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 17:13:54,982 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 17:13:54,982 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 17:13:54,982 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 17:13:54,982 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 17:14:52,135 - util.base_logger - INFO - Epoch=44 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 17:14:53,585 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 17:14:53,585 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 17:14:53,585 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 17:14:53,585 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 17:14:53,585 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 17:14:57,457 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 17:14:57,458 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 17:14:57,458 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 17:14:57,458 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 17:14:57,458 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 17:15:54,616 - util.base_logger - INFO - Epoch=45 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 17:15:56,119 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 17:15:56,120 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 17:15:56,120 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 17:15:56,120 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 17:15:56,120 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 17:15:59,948 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 17:15:59,949 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 17:15:59,949 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 17:15:59,949 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 17:15:59,949 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 17:16:57,025 - util.base_logger - INFO - Epoch=46 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 17:16:58,556 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 17:16:58,556 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 17:16:58,557 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 17:16:58,557 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 17:16:58,557 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 17:17:02,383 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 17:17:02,383 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 17:17:02,383 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 17:17:02,383 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 17:17:02,383 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 17:17:59,683 - util.base_logger - INFO - Epoch=47 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 17:18:01,149 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 17:18:01,149 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 17:18:01,149 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 17:18:01,149 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 17:18:01,149 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 17:18:04,691 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 17:18:04,691 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 17:18:04,691 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 17:18:04,691 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 17:18:04,691 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 17:19:02,216 - util.base_logger - INFO - Epoch=48 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 17:19:03,620 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 17:19:03,621 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 17:19:03,621 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 17:19:03,621 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 17:19:03,621 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 17:19:07,192 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 17:19:07,192 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 17:19:07,192 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 17:19:07,193 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 17:19:07,193 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
[32m2022-04-28 17:20:04,146 - util.base_logger - INFO - Epoch=49 done. (autoencoder.py:254)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 17:20:05,830 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 17:20:05,830 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 17:20:05,830 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 17:20:05,830 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 17:20:05,830 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
/scicore/home/dokman0000/jabjan00/venv_MA/lib/python3.6/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
[33;20m2022-04-28 17:20:09,381 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-04-28 17:20:09,381 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-04-28 17:20:09,381 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13 already exists (utils.py:25)[0m
[33;20m2022-04-28 17:20:09,382 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE already exists (utils.py:25)[0m
[33;20m2022-04-28 17:20:09,382 - util.base_logger - WARNING - ././out/cluster_50-28-04-2022-16-28-13/CovAE/gray-scale already exists (utils.py:25)[0m
