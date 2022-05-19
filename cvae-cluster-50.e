[33;20m2022-05-13 12:27:09,560 - util.base_logger - WARNING - ./out already exists (utils.py:25)[0m
[32m2022-05-13 12:27:09,565 - util.base_logger - INFO - Created folder ./out/cluster_50-13-05-2022-12-27-09 (utils.py:28)[0m
[32m2022-05-13 12:27:09,567 - util.base_logger - INFO - Program starting in cluster-mode (main.py:56)[0m
[32m2022-05-13 12:27:09,567 - util.base_logger - INFO - STARTING PROGRAM (main.py:63)[0m
[32m2022-05-13 12:27:09,567 - util.base_logger - INFO - Selected parameters: (main.py:64)[0m
[32m2022-05-13 12:27:09,567 - util.base_logger - INFO - {'name': 'cluster_50', 'name_time': 'cluster_50-13-05-2022-12-27-09', 'train': True, 'model': './models/models-autoencodergray-scale.pth', 'letters': ['alpha', 'beta', 'chi', 'delta', 'epsilon', 'eta', 'gamma', 'iota', 'kappa', 'lambda', 'mu', 'nu', 'omega', 'omicron', 'phi', 'pi', 'psi', 'rho', 'sigma', 'tau', 'theta', 'xi', 'ypsilon', 'zeta'], 'logging': 40, 'mode': 'cluster', 'batch_size': 512, 'epochs': 50, 'dimensions': 28, 'tqdm': False, 'processing': 'gray-scale', 'root': 'out/cluster_50-13-05-2022-12-27-09'} (main.py:65)[0m
[32m2022-05-13 12:27:09,567 - util.base_logger - INFO - Adjusted dim to %4=0 224 (autoencoder.py:111)[0m
[32m2022-05-13 12:27:09,605 - util.base_logger - INFO - torch.cuda.is_available()=True (autoencoder.py:114)[0m
[32m2022-05-13 12:27:25,018 - util.base_logger - INFO - len(_trainset)=7113 (autoencoder.py:138)[0m
[32m2022-05-13 12:27:25,018 - util.base_logger - INFO - len(_validset)=2371 (autoencoder.py:139)[0m
[32m2022-05-13 12:27:25,018 - util.base_logger - INFO - len(_testset)=2371 (autoencoder.py:140)[0m
[32m2022-05-13 12:27:25,214 - util.base_logger - INFO - ConvAutoEncoder(
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
) (autoencoder.py:160)[0m
[32m2022-05-13 12:27:37,271 - util.base_logger - INFO - Epoch=0 done. (autoencoder.py:267)[0m
[32m2022-05-13 12:27:47,272 - util.base_logger - INFO - Epoch=1 done. (autoencoder.py:267)[0m
[32m2022-05-13 12:27:57,478 - util.base_logger - INFO - Epoch=2 done. (autoencoder.py:267)[0m
[32m2022-05-13 12:28:07,534 - util.base_logger - INFO - Epoch=3 done. (autoencoder.py:267)[0m
[32m2022-05-13 12:28:17,626 - util.base_logger - INFO - Epoch=4 done. (autoencoder.py:267)[0m
[32m2022-05-13 12:28:27,887 - util.base_logger - INFO - Epoch=5 done. (autoencoder.py:267)[0m
[32m2022-05-13 12:28:38,003 - util.base_logger - INFO - Epoch=6 done. (autoencoder.py:267)[0m
[32m2022-05-13 12:28:48,140 - util.base_logger - INFO - Epoch=7 done. (autoencoder.py:267)[0m
[32m2022-05-13 12:28:58,271 - util.base_logger - INFO - Epoch=8 done. (autoencoder.py:267)[0m
[32m2022-05-13 12:29:08,592 - util.base_logger - INFO - Epoch=9 done. (autoencoder.py:267)[0m
[32m2022-05-13 12:29:18,764 - util.base_logger - INFO - Epoch=10 done. (autoencoder.py:267)[0m
[32m2022-05-13 12:29:28,928 - util.base_logger - INFO - Epoch=11 done. (autoencoder.py:267)[0m
[32m2022-05-13 12:29:39,304 - util.base_logger - INFO - Epoch=12 done. (autoencoder.py:267)[0m
[32m2022-05-13 12:29:49,491 - util.base_logger - INFO - Epoch=13 done. (autoencoder.py:267)[0m
[32m2022-05-13 12:29:59,672 - util.base_logger - INFO - Epoch=14 done. (autoencoder.py:267)[0m
[32m2022-05-13 12:30:10,037 - util.base_logger - INFO - Epoch=15 done. (autoencoder.py:267)[0m
[32m2022-05-13 12:30:20,214 - util.base_logger - INFO - Epoch=16 done. (autoencoder.py:267)[0m
[32m2022-05-13 12:30:30,458 - util.base_logger - INFO - Epoch=17 done. (autoencoder.py:267)[0m
[32m2022-05-13 12:30:40,845 - util.base_logger - INFO - Epoch=18 done. (autoencoder.py:267)[0m
[32m2022-05-13 12:30:51,061 - util.base_logger - INFO - Epoch=19 done. (autoencoder.py:267)[0m
[32m2022-05-13 12:31:01,280 - util.base_logger - INFO - Epoch=20 done. (autoencoder.py:267)[0m
[32m2022-05-13 12:31:11,466 - util.base_logger - INFO - Epoch=21 done. (autoencoder.py:267)[0m
[32m2022-05-13 12:31:21,867 - util.base_logger - INFO - Epoch=22 done. (autoencoder.py:267)[0m
[32m2022-05-13 12:31:32,033 - util.base_logger - INFO - Epoch=23 done. (autoencoder.py:267)[0m
[32m2022-05-13 12:31:42,247 - util.base_logger - INFO - Epoch=24 done. (autoencoder.py:267)[0m
[32m2022-05-13 12:31:52,598 - util.base_logger - INFO - Epoch=25 done. (autoencoder.py:267)[0m
[32m2022-05-13 12:32:02,782 - util.base_logger - INFO - Epoch=26 done. (autoencoder.py:267)[0m
[32m2022-05-13 12:32:13,006 - util.base_logger - INFO - Epoch=27 done. (autoencoder.py:267)[0m
[32m2022-05-13 12:32:23,362 - util.base_logger - INFO - Epoch=28 done. (autoencoder.py:267)[0m
[32m2022-05-13 12:32:33,568 - util.base_logger - INFO - Epoch=29 done. (autoencoder.py:267)[0m
[32m2022-05-13 12:32:43,782 - util.base_logger - INFO - Epoch=30 done. (autoencoder.py:267)[0m
[32m2022-05-13 12:32:54,159 - util.base_logger - INFO - Epoch=31 done. (autoencoder.py:267)[0m
[32m2022-05-13 12:33:04,365 - util.base_logger - INFO - Epoch=32 done. (autoencoder.py:267)[0m
[32m2022-05-13 12:33:14,558 - util.base_logger - INFO - Epoch=33 done. (autoencoder.py:267)[0m
[32m2022-05-13 12:33:24,795 - util.base_logger - INFO - Epoch=34 done. (autoencoder.py:267)[0m
[32m2022-05-13 12:33:35,194 - util.base_logger - INFO - Epoch=35 done. (autoencoder.py:267)[0m
[32m2022-05-13 12:33:45,436 - util.base_logger - INFO - Epoch=36 done. (autoencoder.py:267)[0m
[32m2022-05-13 12:33:55,659 - util.base_logger - INFO - Epoch=37 done. (autoencoder.py:267)[0m
[32m2022-05-13 12:34:06,069 - util.base_logger - INFO - Epoch=38 done. (autoencoder.py:267)[0m
[32m2022-05-13 12:34:16,296 - util.base_logger - INFO - Epoch=39 done. (autoencoder.py:267)[0m
[32m2022-05-13 12:34:26,466 - util.base_logger - INFO - Epoch=40 done. (autoencoder.py:267)[0m
[32m2022-05-13 12:34:36,853 - util.base_logger - INFO - Epoch=41 done. (autoencoder.py:267)[0m
[32m2022-05-13 12:34:47,067 - util.base_logger - INFO - Epoch=42 done. (autoencoder.py:267)[0m
[32m2022-05-13 12:34:57,287 - util.base_logger - INFO - Epoch=43 done. (autoencoder.py:267)[0m
[32m2022-05-13 12:35:07,666 - util.base_logger - INFO - Epoch=44 done. (autoencoder.py:267)[0m
[32m2022-05-13 12:35:17,848 - util.base_logger - INFO - Epoch=45 done. (autoencoder.py:267)[0m
[32m2022-05-13 12:35:28,074 - util.base_logger - INFO - Epoch=46 done. (autoencoder.py:267)[0m
[32m2022-05-13 12:35:38,276 - util.base_logger - INFO - Epoch=47 done. (autoencoder.py:267)[0m
[32m2022-05-13 12:35:48,705 - util.base_logger - INFO - Epoch=48 done. (autoencoder.py:267)[0m
[32m2022-05-13 12:35:58,922 - util.base_logger - INFO - Epoch=49 done. (autoencoder.py:267)[0m
[33;20m2022-05-13 12:35:59,945 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-05-13 12:35:59,945 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-05-13 12:35:59,946 - util.base_logger - WARNING - ././out/cluster_50-13-05-2022-12-27-09 already exists (utils.py:25)[0m
[32m2022-05-13 12:35:59,947 - util.base_logger - INFO - Created folder ././out/cluster_50-13-05-2022-12-27-09/net_eval (utils.py:28)[0m
[33;20m2022-05-13 12:36:00,106 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-05-13 12:36:00,107 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-05-13 12:36:00,107 - util.base_logger - WARNING - ././out/cluster_50-13-05-2022-12-27-09 already exists (utils.py:25)[0m
[33;20m2022-05-13 12:36:00,107 - util.base_logger - WARNING - ././out/cluster_50-13-05-2022-12-27-09/net_eval already exists (utils.py:25)[0m
[33;20m2022-05-13 12:36:00,269 - util.base_logger - WARNING - ./. already exists (utils.py:25)[0m
[33;20m2022-05-13 12:36:00,270 - util.base_logger - WARNING - ././out already exists (utils.py:25)[0m
[33;20m2022-05-13 12:36:00,270 - util.base_logger - WARNING - ././out/cluster_50-13-05-2022-12-27-09 already exists (utils.py:25)[0m
[33;20m2022-05-13 12:36:00,270 - util.base_logger - WARNING - ././out/cluster_50-13-05-2022-12-27-09/net_eval already exists (utils.py:25)[0m
