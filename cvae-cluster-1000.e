[33;20m2022-05-24 14:07:45,740 - util.base_logger - WARNING - ./out already exists (utils.py:27)[0m
[32m2022-05-24 14:07:45,779 - util.base_logger - INFO - Created folder ./out/cluster_config_1000_gpu-2022-05-24-14-07-45 (utils.py:30)[0m
[32m2022-05-24 14:07:45,781 - util.base_logger - INFO - STARTING PROGRAM (main.py:47)[0m
[32m2022-05-24 14:07:45,782 - util.base_logger - INFO - Selected parameters: (main.py:48)[0m
[32m2022-05-24 14:07:45,782 - util.base_logger - INFO - {
    "name": "cluster_config_1000_gpu",
    "name_time": "cluster_config_1000_gpu-2022-05-24-14-07-45",
    "train": true,
    "evaluate": true,
    "model_class": "resnet_AE",
    "model_path": null,
    "letters_to_eval": [
        "alpha",
        "beta"
    ],
    "logging_val": 40,
    "batch_size": 512,
    "epochs": 1000,
    "tqdm": false,
    "root": "out/cluster_config_1000_gpu-2022-05-24-14-07-45"
} (main.py:49)[0m
[32m2022-05-24 14:07:45,782 - util.base_logger - INFO - Adjusted dim to %4=0 224 (train.py:62)[0m
[32m2022-05-24 14:07:45,834 - util.base_logger - INFO - torch.cuda.is_available()=True (train.py:65)[0m
[32m2022-05-24 14:08:16,947 - util.base_logger - INFO - len(_trainset)=7113 (train.py:89)[0m
[32m2022-05-24 14:08:16,947 - util.base_logger - INFO - len(_validset)=2371 (train.py:90)[0m
[32m2022-05-24 14:08:16,947 - util.base_logger - INFO - len(_testset)=2371 (train.py:91)[0m
[32m2022-05-24 14:08:16,948 - util.base_logger - INFO - testloader batchsize=512 (train.py:98)[0m
[32m2022-05-24 14:08:20,099 - util.base_logger - INFO - optimizer:torch.optim.adam (train.py:110)[0m
[32m2022-05-24 14:08:20,099 - util.base_logger - INFO - optimizer defaults:{'lr': 0.0001, 'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay': 0, 'amsgrad': False} (train.py:111)[0m
[32m2022-05-24 14:08:20,099 - util.base_logger - INFO - loss:MSELoss() (train.py:112)[0m
[32m2022-05-24 14:08:20,099 - util.base_logger - INFO - loss defaults:<generator object Module.parameters at 0x2b0ea86086d0> (train.py:113)[0m
[32m2022-05-24 14:08:20,099 - util.base_logger - INFO - result obj : {
    "root": "out/cluster_config_1000_gpu-2022-05-24-14-07-45",
    "name": "results-cluster_config_1000_gpu",
    "model": "autoencoders.resnet_ae",
    "epochs": 1000,
    "batch_size": 512,
    "optimizer": "torch.optim.adam",
    "optimizer_args": {
        "lr": 0.0001,
        "betas": [
            0.9,
            0.999
        ],
        "eps": 1e-08,
        "weight_decay": 0,
        "amsgrad": false
    },
    "loss": "MSELoss()",
    "loss_args": null,
    "train_loss": null,
    "valid_loss": null,
    "test_loss": null
} (train.py:127)[0m
[32m2022-05-24 14:08:28,329 - util.base_logger - INFO - Epoch=0 done. (train.py:226)[0m
[32m2022-05-24 14:08:37,528 - util.base_logger - INFO - Epoch=1 done. (train.py:226)[0m
[32m2022-05-24 14:08:46,841 - util.base_logger - INFO - Epoch=2 done. (train.py:226)[0m
[32m2022-05-24 14:08:56,059 - util.base_logger - INFO - Epoch=3 done. (train.py:226)[0m
[32m2022-05-24 14:09:05,276 - util.base_logger - INFO - Epoch=4 done. (train.py:226)[0m
[32m2022-05-24 14:09:14,661 - util.base_logger - INFO - Epoch=5 done. (train.py:226)[0m
[32m2022-05-24 14:09:23,891 - util.base_logger - INFO - Epoch=6 done. (train.py:226)[0m
[32m2022-05-24 14:09:33,131 - util.base_logger - INFO - Epoch=7 done. (train.py:226)[0m
[32m2022-05-24 14:09:42,414 - util.base_logger - INFO - Epoch=8 done. (train.py:226)[0m
[32m2022-05-24 14:09:51,890 - util.base_logger - INFO - Epoch=9 done. (train.py:226)[0m
[32m2022-05-24 14:10:01,156 - util.base_logger - INFO - Epoch=10 done. (train.py:226)[0m
[32m2022-05-24 14:10:10,437 - util.base_logger - INFO - Epoch=11 done. (train.py:226)[0m
[32m2022-05-24 14:10:19,852 - util.base_logger - INFO - Epoch=12 done. (train.py:226)[0m
[32m2022-05-24 14:10:29,123 - util.base_logger - INFO - Epoch=13 done. (train.py:226)[0m
[32m2022-05-24 14:10:38,418 - util.base_logger - INFO - Epoch=14 done. (train.py:226)[0m
[32m2022-05-24 14:10:47,901 - util.base_logger - INFO - Epoch=15 done. (train.py:226)[0m
[32m2022-05-24 14:10:57,173 - util.base_logger - INFO - Epoch=16 done. (train.py:226)[0m
[32m2022-05-24 14:11:06,456 - util.base_logger - INFO - Epoch=17 done. (train.py:226)[0m
[32m2022-05-24 14:11:15,855 - util.base_logger - INFO - Epoch=18 done. (train.py:226)[0m
[32m2022-05-24 14:11:25,130 - util.base_logger - INFO - Epoch=19 done. (train.py:226)[0m
[32m2022-05-24 14:11:34,446 - util.base_logger - INFO - Epoch=20 done. (train.py:226)[0m
[32m2022-05-24 14:11:43,734 - util.base_logger - INFO - Epoch=21 done. (train.py:226)[0m
[32m2022-05-24 14:11:53,201 - util.base_logger - INFO - Epoch=22 done. (train.py:226)[0m
[32m2022-05-24 14:12:02,504 - util.base_logger - INFO - Epoch=23 done. (train.py:226)[0m
[32m2022-05-24 14:12:11,809 - util.base_logger - INFO - Epoch=24 done. (train.py:226)[0m
[32m2022-05-24 14:12:21,273 - util.base_logger - INFO - Epoch=25 done. (train.py:226)[0m
[32m2022-05-24 14:12:30,560 - util.base_logger - INFO - Epoch=26 done. (train.py:226)[0m
[32m2022-05-24 14:12:39,844 - util.base_logger - INFO - Epoch=27 done. (train.py:226)[0m
[32m2022-05-24 14:12:49,293 - util.base_logger - INFO - Epoch=28 done. (train.py:226)[0m
[32m2022-05-24 14:12:58,585 - util.base_logger - INFO - Epoch=29 done. (train.py:226)[0m
[32m2022-05-24 14:13:07,936 - util.base_logger - INFO - Epoch=30 done. (train.py:226)[0m
[32m2022-05-24 14:13:17,397 - util.base_logger - INFO - Epoch=31 done. (train.py:226)[0m
[32m2022-05-24 14:13:26,660 - util.base_logger - INFO - Epoch=32 done. (train.py:226)[0m
[32m2022-05-24 14:13:35,945 - util.base_logger - INFO - Epoch=33 done. (train.py:226)[0m
[32m2022-05-24 14:13:45,253 - util.base_logger - INFO - Epoch=34 done. (train.py:226)[0m
[32m2022-05-24 14:13:54,719 - util.base_logger - INFO - Epoch=35 done. (train.py:226)[0m
[32m2022-05-24 14:14:04,020 - util.base_logger - INFO - Epoch=36 done. (train.py:226)[0m
[32m2022-05-24 14:14:13,322 - util.base_logger - INFO - Epoch=37 done. (train.py:226)[0m
[32m2022-05-24 14:14:22,767 - util.base_logger - INFO - Epoch=38 done. (train.py:226)[0m
[32m2022-05-24 14:14:32,028 - util.base_logger - INFO - Epoch=39 done. (train.py:226)[0m
[32m2022-05-24 14:14:41,308 - util.base_logger - INFO - Epoch=40 done. (train.py:226)[0m
[32m2022-05-24 14:14:50,819 - util.base_logger - INFO - Epoch=41 done. (train.py:226)[0m
[32m2022-05-24 14:15:00,080 - util.base_logger - INFO - Epoch=42 done. (train.py:226)[0m
[32m2022-05-24 14:15:09,369 - util.base_logger - INFO - Epoch=43 done. (train.py:226)[0m
[32m2022-05-24 14:15:18,831 - util.base_logger - INFO - Epoch=44 done. (train.py:226)[0m
[32m2022-05-24 14:15:28,119 - util.base_logger - INFO - Epoch=45 done. (train.py:226)[0m
[32m2022-05-24 14:15:37,419 - util.base_logger - INFO - Epoch=46 done. (train.py:226)[0m
[32m2022-05-24 14:15:46,745 - util.base_logger - INFO - Epoch=47 done. (train.py:226)[0m
[32m2022-05-24 14:15:56,212 - util.base_logger - INFO - Epoch=48 done. (train.py:226)[0m
[32m2022-05-24 14:16:05,472 - util.base_logger - INFO - Epoch=49 done. (train.py:226)[0m
[32m2022-05-24 14:16:14,736 - util.base_logger - INFO - Epoch=50 done. (train.py:226)[0m
[32m2022-05-24 14:16:24,190 - util.base_logger - INFO - Epoch=51 done. (train.py:226)[0m
[32m2022-05-24 14:16:33,511 - util.base_logger - INFO - Epoch=52 done. (train.py:226)[0m
[32m2022-05-24 14:16:42,790 - util.base_logger - INFO - Epoch=53 done. (train.py:226)[0m
[32m2022-05-24 14:16:52,236 - util.base_logger - INFO - Epoch=54 done. (train.py:226)[0m
[32m2022-05-24 14:17:01,504 - util.base_logger - INFO - Epoch=55 done. (train.py:226)[0m
[32m2022-05-24 14:17:10,817 - util.base_logger - INFO - Epoch=56 done. (train.py:226)[0m
[32m2022-05-24 14:17:20,295 - util.base_logger - INFO - Epoch=57 done. (train.py:226)[0m
[32m2022-05-24 14:17:29,607 - util.base_logger - INFO - Epoch=58 done. (train.py:226)[0m
[32m2022-05-24 14:17:38,884 - util.base_logger - INFO - Epoch=59 done. (train.py:226)[0m
[32m2022-05-24 14:17:48,215 - util.base_logger - INFO - Epoch=60 done. (train.py:226)[0m
[32m2022-05-24 14:17:57,663 - util.base_logger - INFO - Epoch=61 done. (train.py:226)[0m
[32m2022-05-24 14:18:06,977 - util.base_logger - INFO - Epoch=62 done. (train.py:226)[0m
[32m2022-05-24 14:18:16,252 - util.base_logger - INFO - Epoch=63 done. (train.py:226)[0m
[32m2022-05-24 14:18:25,706 - util.base_logger - INFO - Epoch=64 done. (train.py:226)[0m
[32m2022-05-24 14:18:34,988 - util.base_logger - INFO - Epoch=65 done. (train.py:226)[0m
[32m2022-05-24 14:18:44,285 - util.base_logger - INFO - Epoch=66 done. (train.py:226)[0m
[32m2022-05-24 14:18:53,765 - util.base_logger - INFO - Epoch=67 done. (train.py:226)[0m
[32m2022-05-24 14:19:03,081 - util.base_logger - INFO - Epoch=68 done. (train.py:226)[0m
[32m2022-05-24 14:19:12,355 - util.base_logger - INFO - Epoch=69 done. (train.py:226)[0m
[32m2022-05-24 14:19:21,777 - util.base_logger - INFO - Epoch=70 done. (train.py:226)[0m
[32m2022-05-24 14:19:31,084 - util.base_logger - INFO - Epoch=71 done. (train.py:226)[0m
[32m2022-05-24 14:19:40,403 - util.base_logger - INFO - Epoch=72 done. (train.py:226)[0m
[32m2022-05-24 14:19:49,749 - util.base_logger - INFO - Epoch=73 done. (train.py:226)[0m
[32m2022-05-24 14:19:59,212 - util.base_logger - INFO - Epoch=74 done. (train.py:226)[0m
[32m2022-05-24 14:20:08,481 - util.base_logger - INFO - Epoch=75 done. (train.py:226)[0m
[32m2022-05-24 14:20:17,749 - util.base_logger - INFO - Epoch=76 done. (train.py:226)[0m
[32m2022-05-24 14:20:27,143 - util.base_logger - INFO - Epoch=77 done. (train.py:226)[0m
[32m2022-05-24 14:20:36,480 - util.base_logger - INFO - Epoch=78 done. (train.py:226)[0m
[32m2022-05-24 14:20:45,789 - util.base_logger - INFO - Epoch=79 done. (train.py:226)[0m
[32m2022-05-24 14:20:55,222 - util.base_logger - INFO - Epoch=80 done. (train.py:226)[0m
[32m2022-05-24 14:21:04,489 - util.base_logger - INFO - Epoch=81 done. (train.py:226)[0m
[32m2022-05-24 14:21:13,769 - util.base_logger - INFO - Epoch=82 done. (train.py:226)[0m
[32m2022-05-24 14:21:23,257 - util.base_logger - INFO - Epoch=83 done. (train.py:226)[0m
[32m2022-05-24 14:21:32,573 - util.base_logger - INFO - Epoch=84 done. (train.py:226)[0m
[32m2022-05-24 14:21:41,875 - util.base_logger - INFO - Epoch=85 done. (train.py:226)[0m
[32m2022-05-24 14:21:51,161 - util.base_logger - INFO - Epoch=86 done. (train.py:226)[0m
[32m2022-05-24 14:22:00,589 - util.base_logger - INFO - Epoch=87 done. (train.py:226)[0m
[32m2022-05-24 14:22:09,866 - util.base_logger - INFO - Epoch=88 done. (train.py:226)[0m
[32m2022-05-24 14:22:19,222 - util.base_logger - INFO - Epoch=89 done. (train.py:226)[0m
[32m2022-05-24 14:22:28,694 - util.base_logger - INFO - Epoch=90 done. (train.py:226)[0m
[32m2022-05-24 14:22:37,994 - util.base_logger - INFO - Epoch=91 done. (train.py:226)[0m
[32m2022-05-24 14:22:47,284 - util.base_logger - INFO - Epoch=92 done. (train.py:226)[0m
[32m2022-05-24 14:22:56,721 - util.base_logger - INFO - Epoch=93 done. (train.py:226)[0m
[32m2022-05-24 14:23:06,017 - util.base_logger - INFO - Epoch=94 done. (train.py:226)[0m
[32m2022-05-24 14:23:15,328 - util.base_logger - INFO - Epoch=95 done. (train.py:226)[0m
[32m2022-05-24 14:23:24,801 - util.base_logger - INFO - Epoch=96 done. (train.py:226)[0m
[32m2022-05-24 14:23:34,073 - util.base_logger - INFO - Epoch=97 done. (train.py:226)[0m
[32m2022-05-24 14:23:43,378 - util.base_logger - INFO - Epoch=98 done. (train.py:226)[0m
[32m2022-05-24 14:23:52,681 - util.base_logger - INFO - Epoch=99 done. (train.py:226)[0m
[32m2022-05-24 14:24:02,169 - util.base_logger - INFO - Epoch=100 done. (train.py:226)[0m
[32m2022-05-24 14:24:11,472 - util.base_logger - INFO - Epoch=101 done. (train.py:226)[0m
[32m2022-05-24 14:24:20,770 - util.base_logger - INFO - Epoch=102 done. (train.py:226)[0m
[32m2022-05-24 14:24:30,237 - util.base_logger - INFO - Epoch=103 done. (train.py:226)[0m
[32m2022-05-24 14:24:39,557 - util.base_logger - INFO - Epoch=104 done. (train.py:226)[0m
[32m2022-05-24 14:24:48,902 - util.base_logger - INFO - Epoch=105 done. (train.py:226)[0m
[32m2022-05-24 14:24:58,358 - util.base_logger - INFO - Epoch=106 done. (train.py:226)[0m
[32m2022-05-24 14:25:07,634 - util.base_logger - INFO - Epoch=107 done. (train.py:226)[0m
[32m2022-05-24 14:25:16,903 - util.base_logger - INFO - Epoch=108 done. (train.py:226)[0m
[32m2022-05-24 14:25:26,373 - util.base_logger - INFO - Epoch=109 done. (train.py:226)[0m
[32m2022-05-24 14:25:35,703 - util.base_logger - INFO - Epoch=110 done. (train.py:226)[0m
[32m2022-05-24 14:25:55,980 - util.base_logger - INFO - Epoch=111 done. (train.py:226)[0m
[32m2022-05-24 14:26:05,236 - util.base_logger - INFO - Epoch=112 done. (train.py:226)[0m
[32m2022-05-24 14:26:14,700 - util.base_logger - INFO - Epoch=113 done. (train.py:226)[0m
[32m2022-05-24 14:26:23,982 - util.base_logger - INFO - Epoch=114 done. (train.py:226)[0m
[32m2022-05-24 14:26:33,292 - util.base_logger - INFO - Epoch=115 done. (train.py:226)[0m
[32m2022-05-24 14:26:42,748 - util.base_logger - INFO - Epoch=116 done. (train.py:226)[0m
[32m2022-05-24 14:26:52,004 - util.base_logger - INFO - Epoch=117 done. (train.py:226)[0m
[32m2022-05-24 14:27:01,326 - util.base_logger - INFO - Epoch=118 done. (train.py:226)[0m
[32m2022-05-24 14:27:10,760 - util.base_logger - INFO - Epoch=119 done. (train.py:226)[0m
[32m2022-05-24 14:27:20,054 - util.base_logger - INFO - Epoch=120 done. (train.py:226)[0m
[32m2022-05-24 14:27:29,393 - util.base_logger - INFO - Epoch=121 done. (train.py:226)[0m
[32m2022-05-24 14:27:38,866 - util.base_logger - INFO - Epoch=122 done. (train.py:226)[0m
[32m2022-05-24 14:27:48,125 - util.base_logger - INFO - Epoch=123 done. (train.py:226)[0m
[32m2022-05-24 14:27:57,369 - util.base_logger - INFO - Epoch=124 done. (train.py:226)[0m
[32m2022-05-24 14:28:06,677 - util.base_logger - INFO - Epoch=125 done. (train.py:226)[0m
[32m2022-05-24 14:28:16,172 - util.base_logger - INFO - Epoch=126 done. (train.py:226)[0m
[32m2022-05-24 14:28:25,476 - util.base_logger - INFO - Epoch=127 done. (train.py:226)[0m
[32m2022-05-24 14:28:34,771 - util.base_logger - INFO - Epoch=128 done. (train.py:226)[0m
[32m2022-05-24 14:28:44,226 - util.base_logger - INFO - Epoch=129 done. (train.py:226)[0m
[32m2022-05-24 14:28:53,502 - util.base_logger - INFO - Epoch=130 done. (train.py:226)[0m
[32m2022-05-24 14:29:02,781 - util.base_logger - INFO - Epoch=131 done. (train.py:226)[0m
[32m2022-05-24 14:29:12,308 - util.base_logger - INFO - Epoch=132 done. (train.py:226)[0m
[32m2022-05-24 14:29:21,579 - util.base_logger - INFO - Epoch=133 done. (train.py:226)[0m
[32m2022-05-24 14:29:30,896 - util.base_logger - INFO - Epoch=134 done. (train.py:226)[0m
[32m2022-05-24 14:29:40,342 - util.base_logger - INFO - Epoch=135 done. (train.py:226)[0m
[32m2022-05-24 14:29:49,633 - util.base_logger - INFO - Epoch=136 done. (train.py:226)[0m
[32m2022-05-24 14:29:58,975 - util.base_logger - INFO - Epoch=137 done. (train.py:226)[0m
[32m2022-05-24 14:30:08,250 - util.base_logger - INFO - Epoch=138 done. (train.py:226)[0m
[32m2022-05-24 14:30:17,732 - util.base_logger - INFO - Epoch=139 done. (train.py:226)[0m
[32m2022-05-24 14:30:27,031 - util.base_logger - INFO - Epoch=140 done. (train.py:226)[0m
[32m2022-05-24 14:30:36,307 - util.base_logger - INFO - Epoch=141 done. (train.py:226)[0m
[32m2022-05-24 14:30:45,814 - util.base_logger - INFO - Epoch=142 done. (train.py:226)[0m
[32m2022-05-24 14:30:55,098 - util.base_logger - INFO - Epoch=143 done. (train.py:226)[0m
[32m2022-05-24 14:31:04,388 - util.base_logger - INFO - Epoch=144 done. (train.py:226)[0m
[32m2022-05-24 14:31:13,828 - util.base_logger - INFO - Epoch=145 done. (train.py:226)[0m
[32m2022-05-24 14:31:23,126 - util.base_logger - INFO - Epoch=146 done. (train.py:226)[0m
[32m2022-05-24 14:31:32,429 - util.base_logger - INFO - Epoch=147 done. (train.py:226)[0m
[32m2022-05-24 14:31:41,943 - util.base_logger - INFO - Epoch=148 done. (train.py:226)[0m
[32m2022-05-24 14:31:51,229 - util.base_logger - INFO - Epoch=149 done. (train.py:226)[0m
[32m2022-05-24 14:32:00,502 - util.base_logger - INFO - Epoch=150 done. (train.py:226)[0m
[32m2022-05-24 14:32:09,831 - util.base_logger - INFO - Epoch=151 done. (train.py:226)[0m
[32m2022-05-24 14:32:19,312 - util.base_logger - INFO - Epoch=152 done. (train.py:226)[0m
[32m2022-05-24 14:32:28,632 - util.base_logger - INFO - Epoch=153 done. (train.py:226)[0m
[32m2022-05-24 14:32:37,925 - util.base_logger - INFO - Epoch=154 done. (train.py:226)[0m
[32m2022-05-24 14:32:47,391 - util.base_logger - INFO - Epoch=155 done. (train.py:226)[0m
[32m2022-05-24 14:32:56,679 - util.base_logger - INFO - Epoch=156 done. (train.py:226)[0m
[32m2022-05-24 14:33:05,944 - util.base_logger - INFO - Epoch=157 done. (train.py:226)[0m
[32m2022-05-24 14:33:15,488 - util.base_logger - INFO - Epoch=158 done. (train.py:226)[0m
[32m2022-05-24 14:33:24,764 - util.base_logger - INFO - Epoch=159 done. (train.py:226)[0m
[32m2022-05-24 14:33:34,098 - util.base_logger - INFO - Epoch=160 done. (train.py:226)[0m
[32m2022-05-24 14:33:43,550 - util.base_logger - INFO - Epoch=161 done. (train.py:226)[0m
[32m2022-05-24 14:33:52,847 - util.base_logger - INFO - Epoch=162 done. (train.py:226)[0m
[32m2022-05-24 14:34:02,187 - util.base_logger - INFO - Epoch=163 done. (train.py:226)[0m
[32m2022-05-24 14:34:11,506 - util.base_logger - INFO - Epoch=164 done. (train.py:226)[0m
[32m2022-05-24 14:34:21,013 - util.base_logger - INFO - Epoch=165 done. (train.py:226)[0m
[32m2022-05-24 14:34:30,291 - util.base_logger - INFO - Epoch=166 done. (train.py:226)[0m
[32m2022-05-24 14:34:39,579 - util.base_logger - INFO - Epoch=167 done. (train.py:226)[0m
[32m2022-05-24 14:34:49,079 - util.base_logger - INFO - Epoch=168 done. (train.py:226)[0m
[32m2022-05-24 14:34:58,358 - util.base_logger - INFO - Epoch=169 done. (train.py:226)[0m
[32m2022-05-24 14:35:07,659 - util.base_logger - INFO - Epoch=170 done. (train.py:226)[0m
[32m2022-05-24 14:35:17,135 - util.base_logger - INFO - Epoch=171 done. (train.py:226)[0m
[32m2022-05-24 14:35:26,414 - util.base_logger - INFO - Epoch=172 done. (train.py:226)[0m
[32m2022-05-24 14:35:35,724 - util.base_logger - INFO - Epoch=173 done. (train.py:226)[0m
[32m2022-05-24 14:35:45,236 - util.base_logger - INFO - Epoch=174 done. (train.py:226)[0m
[32m2022-05-24 14:35:54,560 - util.base_logger - INFO - Epoch=175 done. (train.py:226)[0m
[32m2022-05-24 14:36:27,161 - util.base_logger - INFO - Epoch=176 done. (train.py:226)[0m
[32m2022-05-24 14:36:36,273 - util.base_logger - INFO - Epoch=177 done. (train.py:226)[0m
[32m2022-05-24 14:36:45,495 - util.base_logger - INFO - Epoch=178 done. (train.py:226)[0m
[32m2022-05-24 14:36:54,583 - util.base_logger - INFO - Epoch=179 done. (train.py:226)[0m
[32m2022-05-24 14:37:03,720 - util.base_logger - INFO - Epoch=180 done. (train.py:226)[0m
[32m2022-05-24 14:37:12,995 - util.base_logger - INFO - Epoch=181 done. (train.py:226)[0m
[32m2022-05-24 14:37:22,093 - util.base_logger - INFO - Epoch=182 done. (train.py:226)[0m
[32m2022-05-24 14:37:31,201 - util.base_logger - INFO - Epoch=183 done. (train.py:226)[0m
[32m2022-05-24 14:37:40,467 - util.base_logger - INFO - Epoch=184 done. (train.py:226)[0m
[32m2022-05-24 14:37:49,629 - util.base_logger - INFO - Epoch=185 done. (train.py:226)[0m
[32m2022-05-24 14:37:58,754 - util.base_logger - INFO - Epoch=186 done. (train.py:226)[0m
[32m2022-05-24 14:38:08,060 - util.base_logger - INFO - Epoch=187 done. (train.py:226)[0m
[32m2022-05-24 14:38:17,140 - util.base_logger - INFO - Epoch=188 done. (train.py:226)[0m
[32m2022-05-24 14:38:26,234 - util.base_logger - INFO - Epoch=189 done. (train.py:226)[0m
[32m2022-05-24 14:38:35,370 - util.base_logger - INFO - Epoch=190 done. (train.py:226)[0m
[32m2022-05-24 14:38:44,679 - util.base_logger - INFO - Epoch=191 done. (train.py:226)[0m
[32m2022-05-24 14:38:53,781 - util.base_logger - INFO - Epoch=192 done. (train.py:226)[0m
[32m2022-05-24 14:39:02,895 - util.base_logger - INFO - Epoch=193 done. (train.py:226)[0m
[32m2022-05-24 14:39:12,150 - util.base_logger - INFO - Epoch=194 done. (train.py:226)[0m
[32m2022-05-24 14:39:21,231 - util.base_logger - INFO - Epoch=195 done. (train.py:226)[0m
[32m2022-05-24 14:39:30,368 - util.base_logger - INFO - Epoch=196 done. (train.py:226)[0m
[32m2022-05-24 14:39:39,684 - util.base_logger - INFO - Epoch=197 done. (train.py:226)[0m
[32m2022-05-24 14:39:48,812 - util.base_logger - INFO - Epoch=198 done. (train.py:226)[0m
[32m2022-05-24 14:39:57,899 - util.base_logger - INFO - Epoch=199 done. (train.py:226)[0m
[32m2022-05-24 14:40:07,177 - util.base_logger - INFO - Epoch=200 done. (train.py:226)[0m
[32m2022-05-24 14:40:16,322 - util.base_logger - INFO - Epoch=201 done. (train.py:226)[0m
[32m2022-05-24 14:40:25,422 - util.base_logger - INFO - Epoch=202 done. (train.py:226)[0m
[32m2022-05-24 14:40:34,545 - util.base_logger - INFO - Epoch=203 done. (train.py:226)[0m
[32m2022-05-24 14:40:43,847 - util.base_logger - INFO - Epoch=204 done. (train.py:226)[0m
[32m2022-05-24 14:40:52,916 - util.base_logger - INFO - Epoch=205 done. (train.py:226)[0m
[32m2022-05-24 14:41:02,034 - util.base_logger - INFO - Epoch=206 done. (train.py:226)[0m
[32m2022-05-24 14:41:11,367 - util.base_logger - INFO - Epoch=207 done. (train.py:226)[0m
[32m2022-05-24 14:41:20,479 - util.base_logger - INFO - Epoch=208 done. (train.py:226)[0m
[32m2022-05-24 14:41:29,597 - util.base_logger - INFO - Epoch=209 done. (train.py:226)[0m
[32m2022-05-24 14:41:38,880 - util.base_logger - INFO - Epoch=210 done. (train.py:226)[0m
[32m2022-05-24 14:41:48,005 - util.base_logger - INFO - Epoch=211 done. (train.py:226)[0m
[32m2022-05-24 14:41:57,135 - util.base_logger - INFO - Epoch=212 done. (train.py:226)[0m
[32m2022-05-24 14:42:06,432 - util.base_logger - INFO - Epoch=213 done. (train.py:226)[0m
[32m2022-05-24 14:42:15,559 - util.base_logger - INFO - Epoch=214 done. (train.py:226)[0m
[32m2022-05-24 14:42:24,661 - util.base_logger - INFO - Epoch=215 done. (train.py:226)[0m
[32m2022-05-24 14:42:33,760 - util.base_logger - INFO - Epoch=216 done. (train.py:226)[0m
[32m2022-05-24 14:42:43,112 - util.base_logger - INFO - Epoch=217 done. (train.py:226)[0m
[32m2022-05-24 14:42:52,238 - util.base_logger - INFO - Epoch=218 done. (train.py:226)[0m
[32m2022-05-24 14:43:01,341 - util.base_logger - INFO - Epoch=219 done. (train.py:226)[0m
[32m2022-05-24 14:43:10,600 - util.base_logger - INFO - Epoch=220 done. (train.py:226)[0m
[32m2022-05-24 14:43:19,722 - util.base_logger - INFO - Epoch=221 done. (train.py:226)[0m
[32m2022-05-24 14:43:28,849 - util.base_logger - INFO - Epoch=222 done. (train.py:226)[0m
[32m2022-05-24 14:43:38,168 - util.base_logger - INFO - Epoch=223 done. (train.py:226)[0m
[32m2022-05-24 14:43:47,273 - util.base_logger - INFO - Epoch=224 done. (train.py:226)[0m
[32m2022-05-24 14:43:56,348 - util.base_logger - INFO - Epoch=225 done. (train.py:226)[0m
[32m2022-05-24 14:44:05,577 - util.base_logger - INFO - Epoch=226 done. (train.py:226)[0m
[32m2022-05-24 14:44:14,700 - util.base_logger - INFO - Epoch=227 done. (train.py:226)[0m
[32m2022-05-24 14:44:23,827 - util.base_logger - INFO - Epoch=228 done. (train.py:226)[0m
[32m2022-05-24 14:44:32,945 - util.base_logger - INFO - Epoch=229 done. (train.py:226)[0m
[32m2022-05-24 14:44:42,241 - util.base_logger - INFO - Epoch=230 done. (train.py:226)[0m
[32m2022-05-24 14:44:51,353 - util.base_logger - INFO - Epoch=231 done. (train.py:226)[0m
[32m2022-05-24 14:45:00,464 - util.base_logger - INFO - Epoch=232 done. (train.py:226)[0m
[32m2022-05-24 14:45:09,739 - util.base_logger - INFO - Epoch=233 done. (train.py:226)[0m
[32m2022-05-24 14:45:18,896 - util.base_logger - INFO - Epoch=234 done. (train.py:226)[0m
[32m2022-05-24 14:45:28,038 - util.base_logger - INFO - Epoch=235 done. (train.py:226)[0m
[32m2022-05-24 14:45:37,298 - util.base_logger - INFO - Epoch=236 done. (train.py:226)[0m
[32m2022-05-24 14:45:46,389 - util.base_logger - INFO - Epoch=237 done. (train.py:226)[0m
[32m2022-05-24 14:45:55,518 - util.base_logger - INFO - Epoch=238 done. (train.py:226)[0m
[32m2022-05-24 14:46:04,854 - util.base_logger - INFO - Epoch=239 done. (train.py:226)[0m
[32m2022-05-24 14:46:13,974 - util.base_logger - INFO - Epoch=240 done. (train.py:226)[0m
[32m2022-05-24 14:46:23,088 - util.base_logger - INFO - Epoch=241 done. (train.py:226)[0m
[32m2022-05-24 14:46:32,207 - util.base_logger - INFO - Epoch=242 done. (train.py:226)[0m
[32m2022-05-24 14:46:41,465 - util.base_logger - INFO - Epoch=243 done. (train.py:226)[0m
[32m2022-05-24 14:46:50,576 - util.base_logger - INFO - Epoch=244 done. (train.py:226)[0m
[32m2022-05-24 14:46:59,695 - util.base_logger - INFO - Epoch=245 done. (train.py:226)[0m
[32m2022-05-24 14:47:08,980 - util.base_logger - INFO - Epoch=246 done. (train.py:226)[0m
[32m2022-05-24 14:47:18,071 - util.base_logger - INFO - Epoch=247 done. (train.py:226)[0m
[32m2022-05-24 14:47:27,179 - util.base_logger - INFO - Epoch=248 done. (train.py:226)[0m
[32m2022-05-24 14:47:36,444 - util.base_logger - INFO - Epoch=249 done. (train.py:226)[0m
[32m2022-05-24 14:47:45,605 - util.base_logger - INFO - Epoch=250 done. (train.py:226)[0m
[32m2022-05-24 14:47:54,724 - util.base_logger - INFO - Epoch=251 done. (train.py:226)[0m
[32m2022-05-24 14:48:04,038 - util.base_logger - INFO - Epoch=252 done. (train.py:226)[0m
[32m2022-05-24 14:48:13,159 - util.base_logger - INFO - Epoch=253 done. (train.py:226)[0m
[32m2022-05-24 14:48:22,278 - util.base_logger - INFO - Epoch=254 done. (train.py:226)[0m
[32m2022-05-24 14:48:31,409 - util.base_logger - INFO - Epoch=255 done. (train.py:226)[0m
[32m2022-05-24 14:48:40,736 - util.base_logger - INFO - Epoch=256 done. (train.py:226)[0m
[32m2022-05-24 14:48:49,832 - util.base_logger - INFO - Epoch=257 done. (train.py:226)[0m
[32m2022-05-24 14:48:58,933 - util.base_logger - INFO - Epoch=258 done. (train.py:226)[0m
[32m2022-05-24 14:49:08,167 - util.base_logger - INFO - Epoch=259 done. (train.py:226)[0m
[32m2022-05-24 14:49:17,258 - util.base_logger - INFO - Epoch=260 done. (train.py:226)[0m
[32m2022-05-24 14:49:26,404 - util.base_logger - INFO - Epoch=261 done. (train.py:226)[0m
[32m2022-05-24 14:49:35,717 - util.base_logger - INFO - Epoch=262 done. (train.py:226)[0m
[32m2022-05-24 14:49:44,826 - util.base_logger - INFO - Epoch=263 done. (train.py:226)[0m
[32m2022-05-24 14:49:53,937 - util.base_logger - INFO - Epoch=264 done. (train.py:226)[0m
[32m2022-05-24 14:50:03,202 - util.base_logger - INFO - Epoch=265 done. (train.py:226)[0m
[32m2022-05-24 14:50:12,337 - util.base_logger - INFO - Epoch=266 done. (train.py:226)[0m
[32m2022-05-24 14:50:21,489 - util.base_logger - INFO - Epoch=267 done. (train.py:226)[0m
[32m2022-05-24 14:50:30,621 - util.base_logger - INFO - Epoch=268 done. (train.py:226)[0m
[32m2022-05-24 14:50:39,925 - util.base_logger - INFO - Epoch=269 done. (train.py:226)[0m
[32m2022-05-24 14:50:49,052 - util.base_logger - INFO - Epoch=270 done. (train.py:226)[0m
[32m2022-05-24 14:50:58,177 - util.base_logger - INFO - Epoch=271 done. (train.py:226)[0m
[32m2022-05-24 14:51:07,468 - util.base_logger - INFO - Epoch=272 done. (train.py:226)[0m
[32m2022-05-24 14:51:16,595 - util.base_logger - INFO - Epoch=273 done. (train.py:226)[0m
[32m2022-05-24 14:51:25,711 - util.base_logger - INFO - Epoch=274 done. (train.py:226)[0m
[32m2022-05-24 14:51:34,964 - util.base_logger - INFO - Epoch=275 done. (train.py:226)[0m
[32m2022-05-24 14:51:44,078 - util.base_logger - INFO - Epoch=276 done. (train.py:226)[0m
[32m2022-05-24 14:51:53,236 - util.base_logger - INFO - Epoch=277 done. (train.py:226)[0m
[32m2022-05-24 14:52:02,568 - util.base_logger - INFO - Epoch=278 done. (train.py:226)[0m
[32m2022-05-24 14:52:11,680 - util.base_logger - INFO - Epoch=279 done. (train.py:226)[0m
[32m2022-05-24 14:52:20,808 - util.base_logger - INFO - Epoch=280 done. (train.py:226)[0m
[32m2022-05-24 14:52:29,955 - util.base_logger - INFO - Epoch=281 done. (train.py:226)[0m
[32m2022-05-24 14:52:39,263 - util.base_logger - INFO - Epoch=282 done. (train.py:226)[0m
[32m2022-05-24 14:52:48,407 - util.base_logger - INFO - Epoch=283 done. (train.py:226)[0m
[32m2022-05-24 14:52:57,524 - util.base_logger - INFO - Epoch=284 done. (train.py:226)[0m
[32m2022-05-24 14:53:06,811 - util.base_logger - INFO - Epoch=285 done. (train.py:226)[0m
[32m2022-05-24 14:53:15,898 - util.base_logger - INFO - Epoch=286 done. (train.py:226)[0m
[32m2022-05-24 14:53:24,995 - util.base_logger - INFO - Epoch=287 done. (train.py:226)[0m
[32m2022-05-24 14:53:34,317 - util.base_logger - INFO - Epoch=288 done. (train.py:226)[0m
[32m2022-05-24 14:53:43,430 - util.base_logger - INFO - Epoch=289 done. (train.py:226)[0m
[32m2022-05-24 14:53:52,535 - util.base_logger - INFO - Epoch=290 done. (train.py:226)[0m
[32m2022-05-24 14:54:01,787 - util.base_logger - INFO - Epoch=291 done. (train.py:226)[0m
[32m2022-05-24 14:54:10,924 - util.base_logger - INFO - Epoch=292 done. (train.py:226)[0m
[32m2022-05-24 14:54:20,058 - util.base_logger - INFO - Epoch=293 done. (train.py:226)[0m
[32m2022-05-24 14:54:29,234 - util.base_logger - INFO - Epoch=294 done. (train.py:226)[0m
[32m2022-05-24 14:54:38,553 - util.base_logger - INFO - Epoch=295 done. (train.py:226)[0m
[32m2022-05-24 14:54:47,679 - util.base_logger - INFO - Epoch=296 done. (train.py:226)[0m
[32m2022-05-24 14:54:56,808 - util.base_logger - INFO - Epoch=297 done. (train.py:226)[0m
[32m2022-05-24 14:55:06,086 - util.base_logger - INFO - Epoch=298 done. (train.py:226)[0m
[32m2022-05-24 14:55:15,249 - util.base_logger - INFO - Epoch=299 done. (train.py:226)[0m
[32m2022-05-24 14:55:24,392 - util.base_logger - INFO - Epoch=300 done. (train.py:226)[0m
[32m2022-05-24 14:55:33,681 - util.base_logger - INFO - Epoch=301 done. (train.py:226)[0m
[32m2022-05-24 14:55:42,775 - util.base_logger - INFO - Epoch=302 done. (train.py:226)[0m
[32m2022-05-24 14:55:51,881 - util.base_logger - INFO - Epoch=303 done. (train.py:226)[0m
[32m2022-05-24 14:56:01,185 - util.base_logger - INFO - Epoch=304 done. (train.py:226)[0m
[32m2022-05-24 14:56:10,335 - util.base_logger - INFO - Epoch=305 done. (train.py:226)[0m
[32m2022-05-24 14:56:19,439 - util.base_logger - INFO - Epoch=306 done. (train.py:226)[0m
[32m2022-05-24 14:56:28,569 - util.base_logger - INFO - Epoch=307 done. (train.py:226)[0m
[32m2022-05-24 14:56:37,863 - util.base_logger - INFO - Epoch=308 done. (train.py:226)[0m
[32m2022-05-24 14:56:46,976 - util.base_logger - INFO - Epoch=309 done. (train.py:226)[0m
[32m2022-05-24 14:56:56,137 - util.base_logger - INFO - Epoch=310 done. (train.py:226)[0m
[32m2022-05-24 14:57:05,440 - util.base_logger - INFO - Epoch=311 done. (train.py:226)[0m
[32m2022-05-24 14:57:14,555 - util.base_logger - INFO - Epoch=312 done. (train.py:226)[0m
[32m2022-05-24 14:57:23,680 - util.base_logger - INFO - Epoch=313 done. (train.py:226)[0m
[32m2022-05-24 14:57:32,965 - util.base_logger - INFO - Epoch=314 done. (train.py:226)[0m
[32m2022-05-24 14:57:42,116 - util.base_logger - INFO - Epoch=315 done. (train.py:226)[0m
[32m2022-05-24 14:57:51,230 - util.base_logger - INFO - Epoch=316 done. (train.py:226)[0m
[32m2022-05-24 14:58:00,496 - util.base_logger - INFO - Epoch=317 done. (train.py:226)[0m
[32m2022-05-24 14:58:09,591 - util.base_logger - INFO - Epoch=318 done. (train.py:226)[0m
[32m2022-05-24 14:58:18,703 - util.base_logger - INFO - Epoch=319 done. (train.py:226)[0m
[32m2022-05-24 14:58:27,836 - util.base_logger - INFO - Epoch=320 done. (train.py:226)[0m
[32m2022-05-24 14:58:37,143 - util.base_logger - INFO - Epoch=321 done. (train.py:226)[0m
[32m2022-05-24 14:58:46,236 - util.base_logger - INFO - Epoch=322 done. (train.py:226)[0m
[32m2022-05-24 14:58:55,349 - util.base_logger - INFO - Epoch=323 done. (train.py:226)[0m
[32m2022-05-24 14:59:04,610 - util.base_logger - INFO - Epoch=324 done. (train.py:226)[0m
[32m2022-05-24 14:59:13,736 - util.base_logger - INFO - Epoch=325 done. (train.py:226)[0m
[32m2022-05-24 14:59:22,911 - util.base_logger - INFO - Epoch=326 done. (train.py:226)[0m
[32m2022-05-24 14:59:32,212 - util.base_logger - INFO - Epoch=327 done. (train.py:226)[0m
[32m2022-05-24 14:59:41,337 - util.base_logger - INFO - Epoch=328 done. (train.py:226)[0m
[32m2022-05-24 14:59:50,449 - util.base_logger - INFO - Epoch=329 done. (train.py:226)[0m
[32m2022-05-24 14:59:59,726 - util.base_logger - INFO - Epoch=330 done. (train.py:226)[0m
[32m2022-05-24 15:00:08,873 - util.base_logger - INFO - Epoch=331 done. (train.py:226)[0m
[32m2022-05-24 15:00:18,001 - util.base_logger - INFO - Epoch=332 done. (train.py:226)[0m
[32m2022-05-24 15:00:27,139 - util.base_logger - INFO - Epoch=333 done. (train.py:226)[0m
[32m2022-05-24 15:00:36,446 - util.base_logger - INFO - Epoch=334 done. (train.py:226)[0m
[32m2022-05-24 15:00:45,553 - util.base_logger - INFO - Epoch=335 done. (train.py:226)[0m
[32m2022-05-24 15:00:54,660 - util.base_logger - INFO - Epoch=336 done. (train.py:226)[0m
[32m2022-05-24 15:01:03,991 - util.base_logger - INFO - Epoch=337 done. (train.py:226)[0m
[32m2022-05-24 15:01:13,121 - util.base_logger - INFO - Epoch=338 done. (train.py:226)[0m
[32m2022-05-24 15:01:22,234 - util.base_logger - INFO - Epoch=339 done. (train.py:226)[0m
[32m2022-05-24 15:01:31,500 - util.base_logger - INFO - Epoch=340 done. (train.py:226)[0m
[32m2022-05-24 15:01:40,624 - util.base_logger - INFO - Epoch=341 done. (train.py:226)[0m
[32m2022-05-24 15:01:49,762 - util.base_logger - INFO - Epoch=342 done. (train.py:226)[0m
[32m2022-05-24 15:01:59,080 - util.base_logger - INFO - Epoch=343 done. (train.py:226)[0m
[32m2022-05-24 15:02:08,188 - util.base_logger - INFO - Epoch=344 done. (train.py:226)[0m
[32m2022-05-24 15:02:17,308 - util.base_logger - INFO - Epoch=345 done. (train.py:226)[0m
[32m2022-05-24 15:02:26,443 - util.base_logger - INFO - Epoch=346 done. (train.py:226)[0m
[32m2022-05-24 15:02:35,719 - util.base_logger - INFO - Epoch=347 done. (train.py:226)[0m
[32m2022-05-24 15:02:44,860 - util.base_logger - INFO - Epoch=348 done. (train.py:226)[0m
[32m2022-05-24 15:02:53,967 - util.base_logger - INFO - Epoch=349 done. (train.py:226)[0m
[32m2022-05-24 15:03:03,258 - util.base_logger - INFO - Epoch=350 done. (train.py:226)[0m
[32m2022-05-24 15:03:12,380 - util.base_logger - INFO - Epoch=351 done. (train.py:226)[0m
[32m2022-05-24 15:03:21,495 - util.base_logger - INFO - Epoch=352 done. (train.py:226)[0m
[32m2022-05-24 15:03:30,824 - util.base_logger - INFO - Epoch=353 done. (train.py:226)[0m
[32m2022-05-24 15:03:39,924 - util.base_logger - INFO - Epoch=354 done. (train.py:226)[0m
[32m2022-05-24 15:03:49,077 - util.base_logger - INFO - Epoch=355 done. (train.py:226)[0m
[32m2022-05-24 15:03:58,360 - util.base_logger - INFO - Epoch=356 done. (train.py:226)[0m
[32m2022-05-24 15:04:07,474 - util.base_logger - INFO - Epoch=357 done. (train.py:226)[0m
[32m2022-05-24 15:04:16,592 - util.base_logger - INFO - Epoch=358 done. (train.py:226)[0m
[32m2022-05-24 15:04:25,731 - util.base_logger - INFO - Epoch=359 done. (train.py:226)[0m
[32m2022-05-24 15:04:35,038 - util.base_logger - INFO - Epoch=360 done. (train.py:226)[0m
[32m2022-05-24 15:04:44,166 - util.base_logger - INFO - Epoch=361 done. (train.py:226)[0m
[32m2022-05-24 15:04:53,289 - util.base_logger - INFO - Epoch=362 done. (train.py:226)[0m
[32m2022-05-24 15:05:02,551 - util.base_logger - INFO - Epoch=363 done. (train.py:226)[0m
[32m2022-05-24 15:05:11,682 - util.base_logger - INFO - Epoch=364 done. (train.py:226)[0m
[32m2022-05-24 15:05:20,798 - util.base_logger - INFO - Epoch=365 done. (train.py:226)[0m
[32m2022-05-24 15:05:30,107 - util.base_logger - INFO - Epoch=366 done. (train.py:226)[0m
[32m2022-05-24 15:05:39,226 - util.base_logger - INFO - Epoch=367 done. (train.py:226)[0m
[32m2022-05-24 15:05:48,337 - util.base_logger - INFO - Epoch=368 done. (train.py:226)[0m
[32m2022-05-24 15:05:57,637 - util.base_logger - INFO - Epoch=369 done. (train.py:226)[0m
[32m2022-05-24 15:06:06,765 - util.base_logger - INFO - Epoch=370 done. (train.py:226)[0m
[32m2022-05-24 15:06:15,887 - util.base_logger - INFO - Epoch=371 done. (train.py:226)[0m
[32m2022-05-24 15:06:25,028 - util.base_logger - INFO - Epoch=372 done. (train.py:226)[0m
[32m2022-05-24 15:06:34,290 - util.base_logger - INFO - Epoch=373 done. (train.py:226)[0m
[32m2022-05-24 15:06:43,389 - util.base_logger - INFO - Epoch=374 done. (train.py:226)[0m
[32m2022-05-24 15:06:52,526 - util.base_logger - INFO - Epoch=375 done. (train.py:226)[0m
[32m2022-05-24 15:07:01,846 - util.base_logger - INFO - Epoch=376 done. (train.py:226)[0m
[32m2022-05-24 15:07:10,957 - util.base_logger - INFO - Epoch=377 done. (train.py:226)[0m
[32m2022-05-24 15:07:20,077 - util.base_logger - INFO - Epoch=378 done. (train.py:226)[0m
[32m2022-05-24 15:07:29,374 - util.base_logger - INFO - Epoch=379 done. (train.py:226)[0m
[32m2022-05-24 15:07:38,524 - util.base_logger - INFO - Epoch=380 done. (train.py:226)[0m
[32m2022-05-24 15:07:47,679 - util.base_logger - INFO - Epoch=381 done. (train.py:226)[0m
[32m2022-05-24 15:07:57,009 - util.base_logger - INFO - Epoch=382 done. (train.py:226)[0m
[32m2022-05-24 15:08:06,118 - util.base_logger - INFO - Epoch=383 done. (train.py:226)[0m
[32m2022-05-24 15:08:15,232 - util.base_logger - INFO - Epoch=384 done. (train.py:226)[0m
[32m2022-05-24 15:08:24,354 - util.base_logger - INFO - Epoch=385 done. (train.py:226)[0m
[32m2022-05-24 15:08:33,654 - util.base_logger - INFO - Epoch=386 done. (train.py:226)[0m
[32m2022-05-24 15:08:42,787 - util.base_logger - INFO - Epoch=387 done. (train.py:226)[0m
[32m2022-05-24 15:08:51,904 - util.base_logger - INFO - Epoch=388 done. (train.py:226)[0m
[32m2022-05-24 15:09:01,171 - util.base_logger - INFO - Epoch=389 done. (train.py:226)[0m
[32m2022-05-24 15:09:10,282 - util.base_logger - INFO - Epoch=390 done. (train.py:226)[0m
[32m2022-05-24 15:09:19,402 - util.base_logger - INFO - Epoch=391 done. (train.py:226)[0m
[32m2022-05-24 15:09:28,740 - util.base_logger - INFO - Epoch=392 done. (train.py:226)[0m
[32m2022-05-24 15:09:37,838 - util.base_logger - INFO - Epoch=393 done. (train.py:226)[0m
[32m2022-05-24 15:09:46,952 - util.base_logger - INFO - Epoch=394 done. (train.py:226)[0m
[32m2022-05-24 15:09:56,238 - util.base_logger - INFO - Epoch=395 done. (train.py:226)[0m
[32m2022-05-24 15:10:05,376 - util.base_logger - INFO - Epoch=396 done. (train.py:226)[0m
[32m2022-05-24 15:10:14,504 - util.base_logger - INFO - Epoch=397 done. (train.py:226)[0m
[32m2022-05-24 15:10:23,617 - util.base_logger - INFO - Epoch=398 done. (train.py:226)[0m
[32m2022-05-24 15:10:32,924 - util.base_logger - INFO - Epoch=399 done. (train.py:226)[0m
[32m2022-05-24 15:10:42,054 - util.base_logger - INFO - Epoch=400 done. (train.py:226)[0m
[32m2022-05-24 15:10:51,184 - util.base_logger - INFO - Epoch=401 done. (train.py:226)[0m
[32m2022-05-24 15:11:00,463 - util.base_logger - INFO - Epoch=402 done. (train.py:226)[0m
[32m2022-05-24 15:11:09,608 - util.base_logger - INFO - Epoch=403 done. (train.py:226)[0m
[32m2022-05-24 15:11:18,725 - util.base_logger - INFO - Epoch=404 done. (train.py:226)[0m
[32m2022-05-24 15:11:28,021 - util.base_logger - INFO - Epoch=405 done. (train.py:226)[0m
[32m2022-05-24 15:11:37,158 - util.base_logger - INFO - Epoch=406 done. (train.py:226)[0m
[32m2022-05-24 15:11:46,266 - util.base_logger - INFO - Epoch=407 done. (train.py:226)[0m
[32m2022-05-24 15:11:55,588 - util.base_logger - INFO - Epoch=408 done. (train.py:226)[0m
[32m2022-05-24 15:12:04,712 - util.base_logger - INFO - Epoch=409 done. (train.py:226)[0m
[32m2022-05-24 15:12:13,839 - util.base_logger - INFO - Epoch=410 done. (train.py:226)[0m
[32m2022-05-24 15:12:22,967 - util.base_logger - INFO - Epoch=411 done. (train.py:226)[0m
[32m2022-05-24 15:12:32,220 - util.base_logger - INFO - Epoch=412 done. (train.py:226)[0m
[32m2022-05-24 15:12:41,321 - util.base_logger - INFO - Epoch=413 done. (train.py:226)[0m
[32m2022-05-24 15:12:50,485 - util.base_logger - INFO - Epoch=414 done. (train.py:226)[0m
[32m2022-05-24 15:12:59,783 - util.base_logger - INFO - Epoch=415 done. (train.py:226)[0m
[32m2022-05-24 15:13:08,912 - util.base_logger - INFO - Epoch=416 done. (train.py:226)[0m
[32m2022-05-24 15:13:18,027 - util.base_logger - INFO - Epoch=417 done. (train.py:226)[0m
[32m2022-05-24 15:13:27,276 - util.base_logger - INFO - Epoch=418 done. (train.py:226)[0m
[32m2022-05-24 15:13:36,409 - util.base_logger - INFO - Epoch=419 done. (train.py:226)[0m
[32m2022-05-24 15:13:45,543 - util.base_logger - INFO - Epoch=420 done. (train.py:226)[0m
[32m2022-05-24 15:13:54,844 - util.base_logger - INFO - Epoch=421 done. (train.py:226)[0m
[32m2022-05-24 15:14:03,944 - util.base_logger - INFO - Epoch=422 done. (train.py:226)[0m
[32m2022-05-24 15:14:13,033 - util.base_logger - INFO - Epoch=423 done. (train.py:226)[0m
[32m2022-05-24 15:14:22,160 - util.base_logger - INFO - Epoch=424 done. (train.py:226)[0m
[32m2022-05-24 15:14:31,475 - util.base_logger - INFO - Epoch=425 done. (train.py:226)[0m
[32m2022-05-24 15:14:40,612 - util.base_logger - INFO - Epoch=426 done. (train.py:226)[0m
[32m2022-05-24 15:14:49,743 - util.base_logger - INFO - Epoch=427 done. (train.py:226)[0m
[32m2022-05-24 15:14:59,041 - util.base_logger - INFO - Epoch=428 done. (train.py:226)[0m
[32m2022-05-24 15:15:08,153 - util.base_logger - INFO - Epoch=429 done. (train.py:226)[0m
[32m2022-05-24 15:15:17,290 - util.base_logger - INFO - Epoch=430 done. (train.py:226)[0m
[32m2022-05-24 15:15:26,594 - util.base_logger - INFO - Epoch=431 done. (train.py:226)[0m
[32m2022-05-24 15:15:35,728 - util.base_logger - INFO - Epoch=432 done. (train.py:226)[0m
[32m2022-05-24 15:15:44,824 - util.base_logger - INFO - Epoch=433 done. (train.py:226)[0m
[32m2022-05-24 15:15:54,134 - util.base_logger - INFO - Epoch=434 done. (train.py:226)[0m
[32m2022-05-24 15:16:03,251 - util.base_logger - INFO - Epoch=435 done. (train.py:226)[0m
[32m2022-05-24 15:16:12,372 - util.base_logger - INFO - Epoch=436 done. (train.py:226)[0m
[32m2022-05-24 15:16:21,526 - util.base_logger - INFO - Epoch=437 done. (train.py:226)[0m
[32m2022-05-24 15:16:30,817 - util.base_logger - INFO - Epoch=438 done. (train.py:226)[0m
[32m2022-05-24 15:16:39,932 - util.base_logger - INFO - Epoch=439 done. (train.py:226)[0m
[32m2022-05-24 15:16:49,053 - util.base_logger - INFO - Epoch=440 done. (train.py:226)[0m
[32m2022-05-24 15:16:58,319 - util.base_logger - INFO - Epoch=441 done. (train.py:226)[0m
[32m2022-05-24 15:17:07,423 - util.base_logger - INFO - Epoch=442 done. (train.py:226)[0m
[32m2022-05-24 15:17:16,566 - util.base_logger - INFO - Epoch=443 done. (train.py:226)[0m
[32m2022-05-24 15:17:25,866 - util.base_logger - INFO - Epoch=444 done. (train.py:226)[0m
[32m2022-05-24 15:17:34,959 - util.base_logger - INFO - Epoch=445 done. (train.py:226)[0m
[32m2022-05-24 15:17:44,076 - util.base_logger - INFO - Epoch=446 done. (train.py:226)[0m
[32m2022-05-24 15:17:53,342 - util.base_logger - INFO - Epoch=447 done. (train.py:226)[0m
[32m2022-05-24 15:18:02,504 - util.base_logger - INFO - Epoch=448 done. (train.py:226)[0m
[32m2022-05-24 15:18:11,648 - util.base_logger - INFO - Epoch=449 done. (train.py:226)[0m
[32m2022-05-24 15:18:20,774 - util.base_logger - INFO - Epoch=450 done. (train.py:226)[0m
[32m2022-05-24 15:18:30,069 - util.base_logger - INFO - Epoch=451 done. (train.py:226)[0m
[32m2022-05-24 15:18:39,179 - util.base_logger - INFO - Epoch=452 done. (train.py:226)[0m
[32m2022-05-24 15:18:48,333 - util.base_logger - INFO - Epoch=453 done. (train.py:226)[0m
[32m2022-05-24 15:18:57,654 - util.base_logger - INFO - Epoch=454 done. (train.py:226)[0m
[32m2022-05-24 15:19:06,750 - util.base_logger - INFO - Epoch=455 done. (train.py:226)[0m
[32m2022-05-24 15:19:15,861 - util.base_logger - INFO - Epoch=456 done. (train.py:226)[0m
[32m2022-05-24 15:19:25,119 - util.base_logger - INFO - Epoch=457 done. (train.py:226)[0m
[32m2022-05-24 15:19:34,237 - util.base_logger - INFO - Epoch=458 done. (train.py:226)[0m
[32m2022-05-24 15:19:43,371 - util.base_logger - INFO - Epoch=459 done. (train.py:226)[0m
[32m2022-05-24 15:19:52,689 - util.base_logger - INFO - Epoch=460 done. (train.py:226)[0m
[32m2022-05-24 15:20:01,797 - util.base_logger - INFO - Epoch=461 done. (train.py:226)[0m
[32m2022-05-24 15:20:10,911 - util.base_logger - INFO - Epoch=462 done. (train.py:226)[0m
[32m2022-05-24 15:20:20,034 - util.base_logger - INFO - Epoch=463 done. (train.py:226)[0m
[32m2022-05-24 15:20:29,330 - util.base_logger - INFO - Epoch=464 done. (train.py:226)[0m
[32m2022-05-24 15:20:38,441 - util.base_logger - INFO - Epoch=465 done. (train.py:226)[0m
[32m2022-05-24 15:20:47,602 - util.base_logger - INFO - Epoch=466 done. (train.py:226)[0m
[32m2022-05-24 15:20:56,861 - util.base_logger - INFO - Epoch=467 done. (train.py:226)[0m
[32m2022-05-24 15:21:05,982 - util.base_logger - INFO - Epoch=468 done. (train.py:226)[0m
[32m2022-05-24 15:21:15,078 - util.base_logger - INFO - Epoch=469 done. (train.py:226)[0m
[32m2022-05-24 15:21:24,330 - util.base_logger - INFO - Epoch=470 done. (train.py:226)[0m
[32m2022-05-24 15:21:33,487 - util.base_logger - INFO - Epoch=471 done. (train.py:226)[0m
[32m2022-05-24 15:21:42,608 - util.base_logger - INFO - Epoch=472 done. (train.py:226)[0m
[32m2022-05-24 15:21:51,871 - util.base_logger - INFO - Epoch=473 done. (train.py:226)[0m
[32m2022-05-24 15:22:00,989 - util.base_logger - INFO - Epoch=474 done. (train.py:226)[0m
[32m2022-05-24 15:22:10,125 - util.base_logger - INFO - Epoch=475 done. (train.py:226)[0m
[32m2022-05-24 15:22:19,278 - util.base_logger - INFO - Epoch=476 done. (train.py:226)[0m
[32m2022-05-24 15:22:28,611 - util.base_logger - INFO - Epoch=477 done. (train.py:226)[0m
[32m2022-05-24 15:22:37,710 - util.base_logger - INFO - Epoch=478 done. (train.py:226)[0m
[32m2022-05-24 15:22:46,833 - util.base_logger - INFO - Epoch=479 done. (train.py:226)[0m
[32m2022-05-24 15:22:56,089 - util.base_logger - INFO - Epoch=480 done. (train.py:226)[0m
[32m2022-05-24 15:23:05,183 - util.base_logger - INFO - Epoch=481 done. (train.py:226)[0m
[32m2022-05-24 15:23:14,342 - util.base_logger - INFO - Epoch=482 done. (train.py:226)[0m
[32m2022-05-24 15:23:23,608 - util.base_logger - INFO - Epoch=483 done. (train.py:226)[0m
[32m2022-05-24 15:23:32,728 - util.base_logger - INFO - Epoch=484 done. (train.py:226)[0m
[32m2022-05-24 15:23:41,849 - util.base_logger - INFO - Epoch=485 done. (train.py:226)[0m
[32m2022-05-24 15:23:51,102 - util.base_logger - INFO - Epoch=486 done. (train.py:226)[0m
[32m2022-05-24 15:24:00,266 - util.base_logger - INFO - Epoch=487 done. (train.py:226)[0m
[32m2022-05-24 15:24:09,411 - util.base_logger - INFO - Epoch=488 done. (train.py:226)[0m
[32m2022-05-24 15:24:18,548 - util.base_logger - INFO - Epoch=489 done. (train.py:226)[0m
[32m2022-05-24 15:24:27,873 - util.base_logger - INFO - Epoch=490 done. (train.py:226)[0m
[32m2022-05-24 15:24:36,977 - util.base_logger - INFO - Epoch=491 done. (train.py:226)[0m
[32m2022-05-24 15:24:46,103 - util.base_logger - INFO - Epoch=492 done. (train.py:226)[0m
[32m2022-05-24 15:24:55,426 - util.base_logger - INFO - Epoch=493 done. (train.py:226)[0m
[32m2022-05-24 15:25:04,544 - util.base_logger - INFO - Epoch=494 done. (train.py:226)[0m
[32m2022-05-24 15:25:13,645 - util.base_logger - INFO - Epoch=495 done. (train.py:226)[0m
[32m2022-05-24 15:25:22,923 - util.base_logger - INFO - Epoch=496 done. (train.py:226)[0m
[32m2022-05-24 15:25:32,044 - util.base_logger - INFO - Epoch=497 done. (train.py:226)[0m
[32m2022-05-24 15:25:41,169 - util.base_logger - INFO - Epoch=498 done. (train.py:226)[0m
[32m2022-05-24 15:25:50,507 - util.base_logger - INFO - Epoch=499 done. (train.py:226)[0m
[32m2022-05-24 15:25:59,625 - util.base_logger - INFO - Epoch=500 done. (train.py:226)[0m
[32m2022-05-24 15:26:08,745 - util.base_logger - INFO - Epoch=501 done. (train.py:226)[0m
[32m2022-05-24 15:26:17,877 - util.base_logger - INFO - Epoch=502 done. (train.py:226)[0m
[32m2022-05-24 15:26:27,142 - util.base_logger - INFO - Epoch=503 done. (train.py:226)[0m
[32m2022-05-24 15:26:36,248 - util.base_logger - INFO - Epoch=504 done. (train.py:226)[0m
[32m2022-05-24 15:26:45,379 - util.base_logger - INFO - Epoch=505 done. (train.py:226)[0m
[32m2022-05-24 15:26:54,677 - util.base_logger - INFO - Epoch=506 done. (train.py:226)[0m
[32m2022-05-24 15:27:03,786 - util.base_logger - INFO - Epoch=507 done. (train.py:226)[0m
[32m2022-05-24 15:27:12,922 - util.base_logger - INFO - Epoch=508 done. (train.py:226)[0m
[32m2022-05-24 15:27:22,175 - util.base_logger - INFO - Epoch=509 done. (train.py:226)[0m
[32m2022-05-24 15:27:31,326 - util.base_logger - INFO - Epoch=510 done. (train.py:226)[0m
[32m2022-05-24 15:27:40,429 - util.base_logger - INFO - Epoch=511 done. (train.py:226)[0m
[32m2022-05-24 15:27:49,744 - util.base_logger - INFO - Epoch=512 done. (train.py:226)[0m
[32m2022-05-24 15:27:58,870 - util.base_logger - INFO - Epoch=513 done. (train.py:226)[0m
[32m2022-05-24 15:28:07,966 - util.base_logger - INFO - Epoch=514 done. (train.py:226)[0m
[32m2022-05-24 15:28:17,113 - util.base_logger - INFO - Epoch=515 done. (train.py:226)[0m
[32m2022-05-24 15:28:26,454 - util.base_logger - INFO - Epoch=516 done. (train.py:226)[0m
[32m2022-05-24 15:28:35,556 - util.base_logger - INFO - Epoch=517 done. (train.py:226)[0m
[32m2022-05-24 15:28:44,687 - util.base_logger - INFO - Epoch=518 done. (train.py:226)[0m
[32m2022-05-24 15:28:53,970 - util.base_logger - INFO - Epoch=519 done. (train.py:226)[0m
[32m2022-05-24 15:29:03,068 - util.base_logger - INFO - Epoch=520 done. (train.py:226)[0m
[32m2022-05-24 15:29:12,198 - util.base_logger - INFO - Epoch=521 done. (train.py:226)[0m
[32m2022-05-24 15:29:21,528 - util.base_logger - INFO - Epoch=522 done. (train.py:226)[0m
[32m2022-05-24 15:29:30,643 - util.base_logger - INFO - Epoch=523 done. (train.py:226)[0m
[32m2022-05-24 15:29:39,763 - util.base_logger - INFO - Epoch=524 done. (train.py:226)[0m
[32m2022-05-24 15:29:49,050 - util.base_logger - INFO - Epoch=525 done. (train.py:226)[0m
[32m2022-05-24 15:29:58,155 - util.base_logger - INFO - Epoch=526 done. (train.py:226)[0m
[32m2022-05-24 15:30:07,316 - util.base_logger - INFO - Epoch=527 done. (train.py:226)[0m
[32m2022-05-24 15:30:16,444 - util.base_logger - INFO - Epoch=528 done. (train.py:226)[0m
[32m2022-05-24 15:30:25,735 - util.base_logger - INFO - Epoch=529 done. (train.py:226)[0m
[32m2022-05-24 15:30:34,850 - util.base_logger - INFO - Epoch=530 done. (train.py:226)[0m
[32m2022-05-24 15:30:43,964 - util.base_logger - INFO - Epoch=531 done. (train.py:226)[0m
[32m2022-05-24 15:30:53,252 - util.base_logger - INFO - Epoch=532 done. (train.py:226)[0m
[32m2022-05-24 15:31:02,417 - util.base_logger - INFO - Epoch=533 done. (train.py:226)[0m
[32m2022-05-24 15:31:11,557 - util.base_logger - INFO - Epoch=534 done. (train.py:226)[0m
[32m2022-05-24 15:31:20,840 - util.base_logger - INFO - Epoch=535 done. (train.py:226)[0m
[32m2022-05-24 15:31:29,964 - util.base_logger - INFO - Epoch=536 done. (train.py:226)[0m
[32m2022-05-24 15:31:39,085 - util.base_logger - INFO - Epoch=537 done. (train.py:226)[0m
[32m2022-05-24 15:31:48,397 - util.base_logger - INFO - Epoch=538 done. (train.py:226)[0m
[32m2022-05-24 15:31:57,520 - util.base_logger - INFO - Epoch=539 done. (train.py:226)[0m
[32m2022-05-24 15:32:06,626 - util.base_logger - INFO - Epoch=540 done. (train.py:226)[0m
[32m2022-05-24 15:32:15,761 - util.base_logger - INFO - Epoch=541 done. (train.py:226)[0m
[32m2022-05-24 15:32:25,008 - util.base_logger - INFO - Epoch=542 done. (train.py:226)[0m
[32m2022-05-24 15:32:34,123 - util.base_logger - INFO - Epoch=543 done. (train.py:226)[0m
[32m2022-05-24 15:32:43,259 - util.base_logger - INFO - Epoch=544 done. (train.py:226)[0m
[32m2022-05-24 15:32:52,554 - util.base_logger - INFO - Epoch=545 done. (train.py:226)[0m
[32m2022-05-24 15:33:01,676 - util.base_logger - INFO - Epoch=546 done. (train.py:226)[0m
[32m2022-05-24 15:33:10,822 - util.base_logger - INFO - Epoch=547 done. (train.py:226)[0m
[32m2022-05-24 15:33:20,082 - util.base_logger - INFO - Epoch=548 done. (train.py:226)[0m
[32m2022-05-24 15:33:29,219 - util.base_logger - INFO - Epoch=549 done. (train.py:226)[0m
[32m2022-05-24 15:33:38,366 - util.base_logger - INFO - Epoch=550 done. (train.py:226)[0m
[32m2022-05-24 15:33:47,675 - util.base_logger - INFO - Epoch=551 done. (train.py:226)[0m
[32m2022-05-24 15:33:56,808 - util.base_logger - INFO - Epoch=552 done. (train.py:226)[0m
[32m2022-05-24 15:34:05,938 - util.base_logger - INFO - Epoch=553 done. (train.py:226)[0m
[32m2022-05-24 15:34:15,046 - util.base_logger - INFO - Epoch=554 done. (train.py:226)[0m
[32m2022-05-24 15:34:24,317 - util.base_logger - INFO - Epoch=555 done. (train.py:226)[0m
[32m2022-05-24 15:34:33,454 - util.base_logger - INFO - Epoch=556 done. (train.py:226)[0m
[32m2022-05-24 15:34:42,584 - util.base_logger - INFO - Epoch=557 done. (train.py:226)[0m
[32m2022-05-24 15:34:51,897 - util.base_logger - INFO - Epoch=558 done. (train.py:226)[0m
[32m2022-05-24 15:35:01,020 - util.base_logger - INFO - Epoch=559 done. (train.py:226)[0m
[32m2022-05-24 15:35:10,152 - util.base_logger - INFO - Epoch=560 done. (train.py:226)[0m
[32m2022-05-24 15:35:19,419 - util.base_logger - INFO - Epoch=561 done. (train.py:226)[0m
[32m2022-05-24 15:35:28,591 - util.base_logger - INFO - Epoch=562 done. (train.py:226)[0m
[32m2022-05-24 15:35:37,710 - util.base_logger - INFO - Epoch=563 done. (train.py:226)[0m
[32m2022-05-24 15:35:47,015 - util.base_logger - INFO - Epoch=564 done. (train.py:226)[0m
[32m2022-05-24 15:35:56,124 - util.base_logger - INFO - Epoch=565 done. (train.py:226)[0m
[32m2022-05-24 15:36:05,232 - util.base_logger - INFO - Epoch=566 done. (train.py:226)[0m
[32m2022-05-24 15:36:14,375 - util.base_logger - INFO - Epoch=567 done. (train.py:226)[0m
[32m2022-05-24 15:36:23,708 - util.base_logger - INFO - Epoch=568 done. (train.py:226)[0m
[32m2022-05-24 15:36:32,824 - util.base_logger - INFO - Epoch=569 done. (train.py:226)[0m
[32m2022-05-24 15:36:41,958 - util.base_logger - INFO - Epoch=570 done. (train.py:226)[0m
[32m2022-05-24 15:36:51,243 - util.base_logger - INFO - Epoch=571 done. (train.py:226)[0m
[32m2022-05-24 15:37:00,364 - util.base_logger - INFO - Epoch=572 done. (train.py:226)[0m
[32m2022-05-24 15:37:09,504 - util.base_logger - INFO - Epoch=573 done. (train.py:226)[0m
[32m2022-05-24 15:37:18,825 - util.base_logger - INFO - Epoch=574 done. (train.py:226)[0m
[32m2022-05-24 15:37:27,943 - util.base_logger - INFO - Epoch=575 done. (train.py:226)[0m
[32m2022-05-24 15:37:37,032 - util.base_logger - INFO - Epoch=576 done. (train.py:226)[0m
[32m2022-05-24 15:37:46,298 - util.base_logger - INFO - Epoch=577 done. (train.py:226)[0m
[32m2022-05-24 15:37:55,417 - util.base_logger - INFO - Epoch=578 done. (train.py:226)[0m
[32m2022-05-24 15:38:04,549 - util.base_logger - INFO - Epoch=579 done. (train.py:226)[0m
[32m2022-05-24 15:38:13,682 - util.base_logger - INFO - Epoch=580 done. (train.py:226)[0m
[32m2022-05-24 15:38:22,997 - util.base_logger - INFO - Epoch=581 done. (train.py:226)[0m
[32m2022-05-24 15:38:32,092 - util.base_logger - INFO - Epoch=582 done. (train.py:226)[0m
[32m2022-05-24 15:38:41,221 - util.base_logger - INFO - Epoch=583 done. (train.py:226)[0m
[32m2022-05-24 15:38:50,551 - util.base_logger - INFO - Epoch=584 done. (train.py:226)[0m
[32m2022-05-24 15:38:59,675 - util.base_logger - INFO - Epoch=585 done. (train.py:226)[0m
[32m2022-05-24 15:39:08,800 - util.base_logger - INFO - Epoch=586 done. (train.py:226)[0m
[32m2022-05-24 15:39:18,080 - util.base_logger - INFO - Epoch=587 done. (train.py:226)[0m
[32m2022-05-24 15:39:27,186 - util.base_logger - INFO - Epoch=588 done. (train.py:226)[0m
[32m2022-05-24 15:39:36,297 - util.base_logger - INFO - Epoch=589 done. (train.py:226)[0m
[32m2022-05-24 15:39:45,601 - util.base_logger - INFO - Epoch=590 done. (train.py:226)[0m
[32m2022-05-24 15:39:54,725 - util.base_logger - INFO - Epoch=591 done. (train.py:226)[0m
[32m2022-05-24 15:40:03,855 - util.base_logger - INFO - Epoch=592 done. (train.py:226)[0m
[32m2022-05-24 15:40:12,984 - util.base_logger - INFO - Epoch=593 done. (train.py:226)[0m
[32m2022-05-24 15:40:22,228 - util.base_logger - INFO - Epoch=594 done. (train.py:226)[0m
[32m2022-05-24 15:40:31,347 - util.base_logger - INFO - Epoch=595 done. (train.py:226)[0m
[32m2022-05-24 15:40:40,469 - util.base_logger - INFO - Epoch=596 done. (train.py:226)[0m
[32m2022-05-24 15:40:49,780 - util.base_logger - INFO - Epoch=597 done. (train.py:226)[0m
[32m2022-05-24 15:40:58,888 - util.base_logger - INFO - Epoch=598 done. (train.py:226)[0m
[32m2022-05-24 15:41:08,003 - util.base_logger - INFO - Epoch=599 done. (train.py:226)[0m
[32m2022-05-24 15:41:17,259 - util.base_logger - INFO - Epoch=600 done. (train.py:226)[0m
[32m2022-05-24 15:41:26,390 - util.base_logger - INFO - Epoch=601 done. (train.py:226)[0m
[32m2022-05-24 15:41:35,563 - util.base_logger - INFO - Epoch=602 done. (train.py:226)[0m
[32m2022-05-24 15:41:44,861 - util.base_logger - INFO - Epoch=603 done. (train.py:226)[0m
[32m2022-05-24 15:41:53,964 - util.base_logger - INFO - Epoch=604 done. (train.py:226)[0m
[32m2022-05-24 15:42:03,098 - util.base_logger - INFO - Epoch=605 done. (train.py:226)[0m
[32m2022-05-24 15:42:12,202 - util.base_logger - INFO - Epoch=606 done. (train.py:226)[0m
[32m2022-05-24 15:42:21,501 - util.base_logger - INFO - Epoch=607 done. (train.py:226)[0m
[32m2022-05-24 15:42:30,652 - util.base_logger - INFO - Epoch=608 done. (train.py:226)[0m
[32m2022-05-24 15:42:39,771 - util.base_logger - INFO - Epoch=609 done. (train.py:226)[0m
[32m2022-05-24 15:42:49,062 - util.base_logger - INFO - Epoch=610 done. (train.py:226)[0m
[32m2022-05-24 15:42:58,148 - util.base_logger - INFO - Epoch=611 done. (train.py:226)[0m
[32m2022-05-24 15:43:07,260 - util.base_logger - INFO - Epoch=612 done. (train.py:226)[0m
[32m2022-05-24 15:43:16,544 - util.base_logger - INFO - Epoch=613 done. (train.py:226)[0m
[32m2022-05-24 15:43:25,697 - util.base_logger - INFO - Epoch=614 done. (train.py:226)[0m
[32m2022-05-24 15:43:34,815 - util.base_logger - INFO - Epoch=615 done. (train.py:226)[0m
[32m2022-05-24 15:43:44,103 - util.base_logger - INFO - Epoch=616 done. (train.py:226)[0m
[32m2022-05-24 15:43:53,211 - util.base_logger - INFO - Epoch=617 done. (train.py:226)[0m
[32m2022-05-24 15:44:02,330 - util.base_logger - INFO - Epoch=618 done. (train.py:226)[0m
[32m2022-05-24 15:44:11,460 - util.base_logger - INFO - Epoch=619 done. (train.py:226)[0m
[32m2022-05-24 15:44:20,790 - util.base_logger - INFO - Epoch=620 done. (train.py:226)[0m
[32m2022-05-24 15:44:29,900 - util.base_logger - INFO - Epoch=621 done. (train.py:226)[0m
[32m2022-05-24 15:44:39,045 - util.base_logger - INFO - Epoch=622 done. (train.py:226)[0m
[32m2022-05-24 15:44:48,321 - util.base_logger - INFO - Epoch=623 done. (train.py:226)[0m
[32m2022-05-24 15:44:57,432 - util.base_logger - INFO - Epoch=624 done. (train.py:226)[0m
[32m2022-05-24 15:45:06,539 - util.base_logger - INFO - Epoch=625 done. (train.py:226)[0m
[32m2022-05-24 15:45:15,882 - util.base_logger - INFO - Epoch=626 done. (train.py:226)[0m
[32m2022-05-24 15:45:24,997 - util.base_logger - INFO - Epoch=627 done. (train.py:226)[0m
[32m2022-05-24 15:45:34,128 - util.base_logger - INFO - Epoch=628 done. (train.py:226)[0m
[32m2022-05-24 15:45:43,399 - util.base_logger - INFO - Epoch=629 done. (train.py:226)[0m
[32m2022-05-24 15:45:52,521 - util.base_logger - INFO - Epoch=630 done. (train.py:226)[0m
[32m2022-05-24 15:46:01,652 - util.base_logger - INFO - Epoch=631 done. (train.py:226)[0m
[32m2022-05-24 15:46:10,799 - util.base_logger - INFO - Epoch=632 done. (train.py:226)[0m
[32m2022-05-24 15:46:20,097 - util.base_logger - INFO - Epoch=633 done. (train.py:226)[0m
[32m2022-05-24 15:46:29,204 - util.base_logger - INFO - Epoch=634 done. (train.py:226)[0m
[32m2022-05-24 15:46:38,319 - util.base_logger - INFO - Epoch=635 done. (train.py:226)[0m
[32m2022-05-24 15:46:47,582 - util.base_logger - INFO - Epoch=636 done. (train.py:226)[0m
[32m2022-05-24 15:46:56,688 - util.base_logger - INFO - Epoch=637 done. (train.py:226)[0m
[32m2022-05-24 15:47:05,877 - util.base_logger - INFO - Epoch=638 done. (train.py:226)[0m
[32m2022-05-24 15:47:15,143 - util.base_logger - INFO - Epoch=639 done. (train.py:226)[0m
[32m2022-05-24 15:47:24,263 - util.base_logger - INFO - Epoch=640 done. (train.py:226)[0m
[32m2022-05-24 15:47:33,356 - util.base_logger - INFO - Epoch=641 done. (train.py:226)[0m
[32m2022-05-24 15:47:42,616 - util.base_logger - INFO - Epoch=642 done. (train.py:226)[0m
[32m2022-05-24 15:47:51,743 - util.base_logger - INFO - Epoch=643 done. (train.py:226)[0m
[32m2022-05-24 15:48:00,890 - util.base_logger - INFO - Epoch=644 done. (train.py:226)[0m
[32m2022-05-24 15:48:10,047 - util.base_logger - INFO - Epoch=645 done. (train.py:226)[0m
[32m2022-05-24 15:48:19,325 - util.base_logger - INFO - Epoch=646 done. (train.py:226)[0m
[32m2022-05-24 15:48:28,454 - util.base_logger - INFO - Epoch=647 done. (train.py:226)[0m
[32m2022-05-24 15:48:37,561 - util.base_logger - INFO - Epoch=648 done. (train.py:226)[0m
[32m2022-05-24 15:48:46,854 - util.base_logger - INFO - Epoch=649 done. (train.py:226)[0m
[32m2022-05-24 15:48:56,003 - util.base_logger - INFO - Epoch=650 done. (train.py:226)[0m
[32m2022-05-24 15:49:05,124 - util.base_logger - INFO - Epoch=651 done. (train.py:226)[0m
[32m2022-05-24 15:49:14,402 - util.base_logger - INFO - Epoch=652 done. (train.py:226)[0m
[32m2022-05-24 15:49:23,486 - util.base_logger - INFO - Epoch=653 done. (train.py:226)[0m
[32m2022-05-24 15:49:32,613 - util.base_logger - INFO - Epoch=654 done. (train.py:226)[0m
[32m2022-05-24 15:49:41,898 - util.base_logger - INFO - Epoch=655 done. (train.py:226)[0m
[32m2022-05-24 15:49:51,039 - util.base_logger - INFO - Epoch=656 done. (train.py:226)[0m
[32m2022-05-24 15:50:00,182 - util.base_logger - INFO - Epoch=657 done. (train.py:226)[0m
[32m2022-05-24 15:50:09,308 - util.base_logger - INFO - Epoch=658 done. (train.py:226)[0m
[32m2022-05-24 15:50:18,597 - util.base_logger - INFO - Epoch=659 done. (train.py:226)[0m
[32m2022-05-24 15:50:27,695 - util.base_logger - INFO - Epoch=660 done. (train.py:226)[0m
[32m2022-05-24 15:50:36,807 - util.base_logger - INFO - Epoch=661 done. (train.py:226)[0m
[32m2022-05-24 15:50:46,115 - util.base_logger - INFO - Epoch=662 done. (train.py:226)[0m
[32m2022-05-24 15:50:55,250 - util.base_logger - INFO - Epoch=663 done. (train.py:226)[0m
[32m2022-05-24 15:51:04,399 - util.base_logger - INFO - Epoch=664 done. (train.py:226)[0m
[32m2022-05-24 15:51:13,676 - util.base_logger - INFO - Epoch=665 done. (train.py:226)[0m
[32m2022-05-24 15:51:22,791 - util.base_logger - INFO - Epoch=666 done. (train.py:226)[0m
[32m2022-05-24 15:51:31,892 - util.base_logger - INFO - Epoch=667 done. (train.py:226)[0m
[32m2022-05-24 15:51:41,216 - util.base_logger - INFO - Epoch=668 done. (train.py:226)[0m
[32m2022-05-24 15:51:50,346 - util.base_logger - INFO - Epoch=669 done. (train.py:226)[0m
[32m2022-05-24 15:51:59,467 - util.base_logger - INFO - Epoch=670 done. (train.py:226)[0m
[32m2022-05-24 15:52:08,603 - util.base_logger - INFO - Epoch=671 done. (train.py:226)[0m
[32m2022-05-24 15:52:17,855 - util.base_logger - INFO - Epoch=672 done. (train.py:226)[0m
[32m2022-05-24 15:52:26,971 - util.base_logger - INFO - Epoch=673 done. (train.py:226)[0m
[32m2022-05-24 15:52:36,100 - util.base_logger - INFO - Epoch=674 done. (train.py:226)[0m
[32m2022-05-24 15:52:45,402 - util.base_logger - INFO - Epoch=675 done. (train.py:226)[0m
[32m2022-05-24 15:52:54,493 - util.base_logger - INFO - Epoch=676 done. (train.py:226)[0m
[32m2022-05-24 15:53:03,604 - util.base_logger - INFO - Epoch=677 done. (train.py:226)[0m
[32m2022-05-24 15:53:12,892 - util.base_logger - INFO - Epoch=678 done. (train.py:226)[0m
[32m2022-05-24 15:53:22,009 - util.base_logger - INFO - Epoch=679 done. (train.py:226)[0m
[32m2022-05-24 15:53:31,166 - util.base_logger - INFO - Epoch=680 done. (train.py:226)[0m
[32m2022-05-24 15:53:40,473 - util.base_logger - INFO - Epoch=681 done. (train.py:226)[0m
[32m2022-05-24 15:53:49,610 - util.base_logger - INFO - Epoch=682 done. (train.py:226)[0m
[32m2022-05-24 15:53:58,723 - util.base_logger - INFO - Epoch=683 done. (train.py:226)[0m
[32m2022-05-24 15:54:07,871 - util.base_logger - INFO - Epoch=684 done. (train.py:226)[0m
[32m2022-05-24 15:54:17,166 - util.base_logger - INFO - Epoch=685 done. (train.py:226)[0m
[32m2022-05-24 15:54:26,294 - util.base_logger - INFO - Epoch=686 done. (train.py:226)[0m
[32m2022-05-24 15:54:35,460 - util.base_logger - INFO - Epoch=687 done. (train.py:226)[0m
[32m2022-05-24 15:54:44,757 - util.base_logger - INFO - Epoch=688 done. (train.py:226)[0m
[32m2022-05-24 15:54:53,879 - util.base_logger - INFO - Epoch=689 done. (train.py:226)[0m
[32m2022-05-24 15:55:02,979 - util.base_logger - INFO - Epoch=690 done. (train.py:226)[0m
[32m2022-05-24 15:55:12,259 - util.base_logger - INFO - Epoch=691 done. (train.py:226)[0m
[32m2022-05-24 15:55:21,391 - util.base_logger - INFO - Epoch=692 done. (train.py:226)[0m
[32m2022-05-24 15:55:30,552 - util.base_logger - INFO - Epoch=693 done. (train.py:226)[0m
[32m2022-05-24 15:55:39,849 - util.base_logger - INFO - Epoch=694 done. (train.py:226)[0m
[32m2022-05-24 15:55:48,983 - util.base_logger - INFO - Epoch=695 done. (train.py:226)[0m
[32m2022-05-24 15:55:58,071 - util.base_logger - INFO - Epoch=696 done. (train.py:226)[0m
[32m2022-05-24 15:56:07,185 - util.base_logger - INFO - Epoch=697 done. (train.py:226)[0m
[32m2022-05-24 15:56:16,484 - util.base_logger - INFO - Epoch=698 done. (train.py:226)[0m
[32m2022-05-24 15:56:25,624 - util.base_logger - INFO - Epoch=699 done. (train.py:226)[0m
[32m2022-05-24 15:56:34,752 - util.base_logger - INFO - Epoch=700 done. (train.py:226)[0m
[32m2022-05-24 15:56:44,024 - util.base_logger - INFO - Epoch=701 done. (train.py:226)[0m
[32m2022-05-24 15:56:53,118 - util.base_logger - INFO - Epoch=702 done. (train.py:226)[0m
[32m2022-05-24 15:57:02,246 - util.base_logger - INFO - Epoch=703 done. (train.py:226)[0m
[32m2022-05-24 15:57:11,540 - util.base_logger - INFO - Epoch=704 done. (train.py:226)[0m
[32m2022-05-24 15:57:20,690 - util.base_logger - INFO - Epoch=705 done. (train.py:226)[0m
[32m2022-05-24 15:57:29,803 - util.base_logger - INFO - Epoch=706 done. (train.py:226)[0m
[32m2022-05-24 15:57:39,089 - util.base_logger - INFO - Epoch=707 done. (train.py:226)[0m
[32m2022-05-24 15:57:48,191 - util.base_logger - INFO - Epoch=708 done. (train.py:226)[0m
[32m2022-05-24 15:57:57,306 - util.base_logger - INFO - Epoch=709 done. (train.py:226)[0m
[32m2022-05-24 15:58:06,458 - util.base_logger - INFO - Epoch=710 done. (train.py:226)[0m
[32m2022-05-24 15:58:15,772 - util.base_logger - INFO - Epoch=711 done. (train.py:226)[0m
[32m2022-05-24 15:58:24,881 - util.base_logger - INFO - Epoch=712 done. (train.py:226)[0m
[32m2022-05-24 15:58:33,996 - util.base_logger - INFO - Epoch=713 done. (train.py:226)[0m
[32m2022-05-24 15:58:43,269 - util.base_logger - INFO - Epoch=714 done. (train.py:226)[0m
[32m2022-05-24 15:58:52,415 - util.base_logger - INFO - Epoch=715 done. (train.py:226)[0m
[32m2022-05-24 15:59:01,571 - util.base_logger - INFO - Epoch=716 done. (train.py:226)[0m
[32m2022-05-24 15:59:10,884 - util.base_logger - INFO - Epoch=717 done. (train.py:226)[0m
[32m2022-05-24 15:59:19,978 - util.base_logger - INFO - Epoch=718 done. (train.py:226)[0m
[32m2022-05-24 15:59:29,100 - util.base_logger - INFO - Epoch=719 done. (train.py:226)[0m
[32m2022-05-24 15:59:38,364 - util.base_logger - INFO - Epoch=720 done. (train.py:226)[0m
[32m2022-05-24 15:59:47,486 - util.base_logger - INFO - Epoch=721 done. (train.py:226)[0m
[32m2022-05-24 15:59:56,625 - util.base_logger - INFO - Epoch=722 done. (train.py:226)[0m
[32m2022-05-24 16:00:05,771 - util.base_logger - INFO - Epoch=723 done. (train.py:226)[0m
[32m2022-05-24 16:00:15,082 - util.base_logger - INFO - Epoch=724 done. (train.py:226)[0m
[32m2022-05-24 16:00:24,169 - util.base_logger - INFO - Epoch=725 done. (train.py:226)[0m
[32m2022-05-24 16:00:33,314 - util.base_logger - INFO - Epoch=726 done. (train.py:226)[0m
[32m2022-05-24 16:00:42,570 - util.base_logger - INFO - Epoch=727 done. (train.py:226)[0m
[32m2022-05-24 16:00:51,732 - util.base_logger - INFO - Epoch=728 done. (train.py:226)[0m
[32m2022-05-24 16:01:00,889 - util.base_logger - INFO - Epoch=729 done. (train.py:226)[0m
[32m2022-05-24 16:01:10,194 - util.base_logger - INFO - Epoch=730 done. (train.py:226)[0m
[32m2022-05-24 16:01:19,301 - util.base_logger - INFO - Epoch=731 done. (train.py:226)[0m
[32m2022-05-24 16:01:28,418 - util.base_logger - INFO - Epoch=732 done. (train.py:226)[0m
[32m2022-05-24 16:01:37,673 - util.base_logger - INFO - Epoch=733 done. (train.py:226)[0m
[32m2022-05-24 16:01:46,790 - util.base_logger - INFO - Epoch=734 done. (train.py:226)[0m
[32m2022-05-24 16:01:55,927 - util.base_logger - INFO - Epoch=735 done. (train.py:226)[0m
[32m2022-05-24 16:02:05,062 - util.base_logger - INFO - Epoch=736 done. (train.py:226)[0m
[32m2022-05-24 16:02:14,375 - util.base_logger - INFO - Epoch=737 done. (train.py:226)[0m
[32m2022-05-24 16:02:23,519 - util.base_logger - INFO - Epoch=738 done. (train.py:226)[0m
[32m2022-05-24 16:02:32,641 - util.base_logger - INFO - Epoch=739 done. (train.py:226)[0m
[32m2022-05-24 16:02:41,943 - util.base_logger - INFO - Epoch=740 done. (train.py:226)[0m
[32m2022-05-24 16:02:51,088 - util.base_logger - INFO - Epoch=741 done. (train.py:226)[0m
[32m2022-05-24 16:03:00,216 - util.base_logger - INFO - Epoch=742 done. (train.py:226)[0m
[32m2022-05-24 16:03:09,511 - util.base_logger - INFO - Epoch=743 done. (train.py:226)[0m
[32m2022-05-24 16:03:18,643 - util.base_logger - INFO - Epoch=744 done. (train.py:226)[0m
[32m2022-05-24 16:03:27,766 - util.base_logger - INFO - Epoch=745 done. (train.py:226)[0m
[32m2022-05-24 16:03:37,081 - util.base_logger - INFO - Epoch=746 done. (train.py:226)[0m
[32m2022-05-24 16:03:46,235 - util.base_logger - INFO - Epoch=747 done. (train.py:226)[0m
[32m2022-05-24 16:03:55,333 - util.base_logger - INFO - Epoch=748 done. (train.py:226)[0m
[32m2022-05-24 16:04:04,467 - util.base_logger - INFO - Epoch=749 done. (train.py:226)[0m
[32m2022-05-24 16:04:13,738 - util.base_logger - INFO - Epoch=750 done. (train.py:226)[0m
[32m2022-05-24 16:04:22,856 - util.base_logger - INFO - Epoch=751 done. (train.py:226)[0m
[32m2022-05-24 16:04:32,009 - util.base_logger - INFO - Epoch=752 done. (train.py:226)[0m
[32m2022-05-24 16:04:41,349 - util.base_logger - INFO - Epoch=753 done. (train.py:226)[0m
[32m2022-05-24 16:04:50,488 - util.base_logger - INFO - Epoch=754 done. (train.py:226)[0m
[32m2022-05-24 16:04:59,607 - util.base_logger - INFO - Epoch=755 done. (train.py:226)[0m
[32m2022-05-24 16:05:08,898 - util.base_logger - INFO - Epoch=756 done. (train.py:226)[0m
[32m2022-05-24 16:05:18,003 - util.base_logger - INFO - Epoch=757 done. (train.py:226)[0m
[32m2022-05-24 16:05:27,145 - util.base_logger - INFO - Epoch=758 done. (train.py:226)[0m
[32m2022-05-24 16:05:36,510 - util.base_logger - INFO - Epoch=759 done. (train.py:226)[0m
[32m2022-05-24 16:05:45,639 - util.base_logger - INFO - Epoch=760 done. (train.py:226)[0m
[32m2022-05-24 16:05:54,763 - util.base_logger - INFO - Epoch=761 done. (train.py:226)[0m
[32m2022-05-24 16:06:03,865 - util.base_logger - INFO - Epoch=762 done. (train.py:226)[0m
[32m2022-05-24 16:06:13,147 - util.base_logger - INFO - Epoch=763 done. (train.py:226)[0m
[32m2022-05-24 16:06:22,274 - util.base_logger - INFO - Epoch=764 done. (train.py:226)[0m
[32m2022-05-24 16:06:31,427 - util.base_logger - INFO - Epoch=765 done. (train.py:226)[0m
[32m2022-05-24 16:06:40,737 - util.base_logger - INFO - Epoch=766 done. (train.py:226)[0m
[32m2022-05-24 16:06:49,849 - util.base_logger - INFO - Epoch=767 done. (train.py:226)[0m
[32m2022-05-24 16:06:58,989 - util.base_logger - INFO - Epoch=768 done. (train.py:226)[0m
[32m2022-05-24 16:07:08,247 - util.base_logger - INFO - Epoch=769 done. (train.py:226)[0m
[32m2022-05-24 16:07:17,393 - util.base_logger - INFO - Epoch=770 done. (train.py:226)[0m
[32m2022-05-24 16:07:26,525 - util.base_logger - INFO - Epoch=771 done. (train.py:226)[0m
[32m2022-05-24 16:07:35,833 - util.base_logger - INFO - Epoch=772 done. (train.py:226)[0m
[32m2022-05-24 16:07:44,949 - util.base_logger - INFO - Epoch=773 done. (train.py:226)[0m
[32m2022-05-24 16:07:54,075 - util.base_logger - INFO - Epoch=774 done. (train.py:226)[0m
[32m2022-05-24 16:08:03,224 - util.base_logger - INFO - Epoch=775 done. (train.py:226)[0m
[32m2022-05-24 16:08:12,534 - util.base_logger - INFO - Epoch=776 done. (train.py:226)[0m
[32m2022-05-24 16:08:21,671 - util.base_logger - INFO - Epoch=777 done. (train.py:226)[0m
[32m2022-05-24 16:08:30,790 - util.base_logger - INFO - Epoch=778 done. (train.py:226)[0m
[32m2022-05-24 16:08:40,083 - util.base_logger - INFO - Epoch=779 done. (train.py:226)[0m
[32m2022-05-24 16:08:49,214 - util.base_logger - INFO - Epoch=780 done. (train.py:226)[0m
[32m2022-05-24 16:08:58,349 - util.base_logger - INFO - Epoch=781 done. (train.py:226)[0m
[32m2022-05-24 16:09:07,646 - util.base_logger - INFO - Epoch=782 done. (train.py:226)[0m
[32m2022-05-24 16:09:16,783 - util.base_logger - INFO - Epoch=783 done. (train.py:226)[0m
[32m2022-05-24 16:09:25,908 - util.base_logger - INFO - Epoch=784 done. (train.py:226)[0m
[32m2022-05-24 16:09:35,181 - util.base_logger - INFO - Epoch=785 done. (train.py:226)[0m
[32m2022-05-24 16:09:44,294 - util.base_logger - INFO - Epoch=786 done. (train.py:226)[0m
[32m2022-05-24 16:09:53,404 - util.base_logger - INFO - Epoch=787 done. (train.py:226)[0m
[32m2022-05-24 16:10:02,525 - util.base_logger - INFO - Epoch=788 done. (train.py:226)[0m
[32m2022-05-24 16:10:11,901 - util.base_logger - INFO - Epoch=789 done. (train.py:226)[0m
[32m2022-05-24 16:10:21,008 - util.base_logger - INFO - Epoch=790 done. (train.py:226)[0m
[32m2022-05-24 16:10:30,150 - util.base_logger - INFO - Epoch=791 done. (train.py:226)[0m
[32m2022-05-24 16:10:39,395 - util.base_logger - INFO - Epoch=792 done. (train.py:226)[0m
[32m2022-05-24 16:10:48,522 - util.base_logger - INFO - Epoch=793 done. (train.py:226)[0m
[32m2022-05-24 16:10:57,635 - util.base_logger - INFO - Epoch=794 done. (train.py:226)[0m
[32m2022-05-24 16:11:06,967 - util.base_logger - INFO - Epoch=795 done. (train.py:226)[0m
[32m2022-05-24 16:11:16,065 - util.base_logger - INFO - Epoch=796 done. (train.py:226)[0m
[32m2022-05-24 16:11:25,185 - util.base_logger - INFO - Epoch=797 done. (train.py:226)[0m
[32m2022-05-24 16:11:34,437 - util.base_logger - INFO - Epoch=798 done. (train.py:226)[0m
[32m2022-05-24 16:11:43,565 - util.base_logger - INFO - Epoch=799 done. (train.py:226)[0m
[32m2022-05-24 16:11:52,667 - util.base_logger - INFO - Epoch=800 done. (train.py:226)[0m
[32m2022-05-24 16:12:01,853 - util.base_logger - INFO - Epoch=801 done. (train.py:226)[0m
[32m2022-05-24 16:12:11,149 - util.base_logger - INFO - Epoch=802 done. (train.py:226)[0m
[32m2022-05-24 16:12:20,275 - util.base_logger - INFO - Epoch=803 done. (train.py:226)[0m
[32m2022-05-24 16:12:29,387 - util.base_logger - INFO - Epoch=804 done. (train.py:226)[0m
[32m2022-05-24 16:12:38,686 - util.base_logger - INFO - Epoch=805 done. (train.py:226)[0m
[32m2022-05-24 16:12:47,798 - util.base_logger - INFO - Epoch=806 done. (train.py:226)[0m
[32m2022-05-24 16:12:56,967 - util.base_logger - INFO - Epoch=807 done. (train.py:226)[0m
[32m2022-05-24 16:13:06,254 - util.base_logger - INFO - Epoch=808 done. (train.py:226)[0m
[32m2022-05-24 16:13:15,378 - util.base_logger - INFO - Epoch=809 done. (train.py:226)[0m
[32m2022-05-24 16:13:24,487 - util.base_logger - INFO - Epoch=810 done. (train.py:226)[0m
[32m2022-05-24 16:13:33,771 - util.base_logger - INFO - Epoch=811 done. (train.py:226)[0m
[32m2022-05-24 16:13:42,891 - util.base_logger - INFO - Epoch=812 done. (train.py:226)[0m
[32m2022-05-24 16:13:52,084 - util.base_logger - INFO - Epoch=813 done. (train.py:226)[0m
[32m2022-05-24 16:14:01,206 - util.base_logger - INFO - Epoch=814 done. (train.py:226)[0m
[32m2022-05-24 16:14:10,505 - util.base_logger - INFO - Epoch=815 done. (train.py:226)[0m
[32m2022-05-24 16:14:19,597 - util.base_logger - INFO - Epoch=816 done. (train.py:226)[0m
[32m2022-05-24 16:14:28,741 - util.base_logger - INFO - Epoch=817 done. (train.py:226)[0m
[32m2022-05-24 16:14:37,994 - util.base_logger - INFO - Epoch=818 done. (train.py:226)[0m
[32m2022-05-24 16:14:47,122 - util.base_logger - INFO - Epoch=819 done. (train.py:226)[0m
[32m2022-05-24 16:14:56,238 - util.base_logger - INFO - Epoch=820 done. (train.py:226)[0m
[32m2022-05-24 16:15:05,526 - util.base_logger - INFO - Epoch=821 done. (train.py:226)[0m
[32m2022-05-24 16:15:14,607 - util.base_logger - INFO - Epoch=822 done. (train.py:226)[0m
[32m2022-05-24 16:15:23,743 - util.base_logger - INFO - Epoch=823 done. (train.py:226)[0m
[32m2022-05-24 16:15:33,007 - util.base_logger - INFO - Epoch=824 done. (train.py:226)[0m
[32m2022-05-24 16:15:42,163 - util.base_logger - INFO - Epoch=825 done. (train.py:226)[0m
[32m2022-05-24 16:15:51,288 - util.base_logger - INFO - Epoch=826 done. (train.py:226)[0m
[32m2022-05-24 16:16:00,434 - util.base_logger - INFO - Epoch=827 done. (train.py:226)[0m
[32m2022-05-24 16:16:09,717 - util.base_logger - INFO - Epoch=828 done. (train.py:226)[0m
[32m2022-05-24 16:16:18,843 - util.base_logger - INFO - Epoch=829 done. (train.py:226)[0m
[32m2022-05-24 16:16:27,952 - util.base_logger - INFO - Epoch=830 done. (train.py:226)[0m
[32m2022-05-24 16:16:37,303 - util.base_logger - INFO - Epoch=831 done. (train.py:226)[0m
[32m2022-05-24 16:16:46,443 - util.base_logger - INFO - Epoch=832 done. (train.py:226)[0m
[32m2022-05-24 16:16:55,554 - util.base_logger - INFO - Epoch=833 done. (train.py:226)[0m
[32m2022-05-24 16:17:04,807 - util.base_logger - INFO - Epoch=834 done. (train.py:226)[0m
[32m2022-05-24 16:17:13,918 - util.base_logger - INFO - Epoch=835 done. (train.py:226)[0m
[32m2022-05-24 16:17:23,029 - util.base_logger - INFO - Epoch=836 done. (train.py:226)[0m
[32m2022-05-24 16:17:32,386 - util.base_logger - INFO - Epoch=837 done. (train.py:226)[0m
[32m2022-05-24 16:17:41,494 - util.base_logger - INFO - Epoch=838 done. (train.py:226)[0m
[32m2022-05-24 16:17:50,625 - util.base_logger - INFO - Epoch=839 done. (train.py:226)[0m
[32m2022-05-24 16:17:59,776 - util.base_logger - INFO - Epoch=840 done. (train.py:226)[0m
[32m2022-05-24 16:18:09,078 - util.base_logger - INFO - Epoch=841 done. (train.py:226)[0m
[32m2022-05-24 16:18:18,183 - util.base_logger - INFO - Epoch=842 done. (train.py:226)[0m
[32m2022-05-24 16:18:27,346 - util.base_logger - INFO - Epoch=843 done. (train.py:226)[0m
[32m2022-05-24 16:18:36,645 - util.base_logger - INFO - Epoch=844 done. (train.py:226)[0m
[32m2022-05-24 16:18:45,759 - util.base_logger - INFO - Epoch=845 done. (train.py:226)[0m
[32m2022-05-24 16:18:54,857 - util.base_logger - INFO - Epoch=846 done. (train.py:226)[0m
[32m2022-05-24 16:19:04,145 - util.base_logger - INFO - Epoch=847 done. (train.py:226)[0m
[32m2022-05-24 16:19:13,243 - util.base_logger - INFO - Epoch=848 done. (train.py:226)[0m
[32m2022-05-24 16:19:22,397 - util.base_logger - INFO - Epoch=849 done. (train.py:226)[0m
[32m2022-05-24 16:19:31,689 - util.base_logger - INFO - Epoch=850 done. (train.py:226)[0m
[32m2022-05-24 16:19:40,808 - util.base_logger - INFO - Epoch=851 done. (train.py:226)[0m
[32m2022-05-24 16:19:49,917 - util.base_logger - INFO - Epoch=852 done. (train.py:226)[0m
[32m2022-05-24 16:19:59,078 - util.base_logger - INFO - Epoch=853 done. (train.py:226)[0m
[32m2022-05-24 16:20:08,346 - util.base_logger - INFO - Epoch=854 done. (train.py:226)[0m
[32m2022-05-24 16:20:17,506 - util.base_logger - INFO - Epoch=855 done. (train.py:226)[0m
[32m2022-05-24 16:20:26,626 - util.base_logger - INFO - Epoch=856 done. (train.py:226)[0m
[32m2022-05-24 16:20:35,940 - util.base_logger - INFO - Epoch=857 done. (train.py:226)[0m
[32m2022-05-24 16:20:45,043 - util.base_logger - INFO - Epoch=858 done. (train.py:226)[0m
[32m2022-05-24 16:20:54,171 - util.base_logger - INFO - Epoch=859 done. (train.py:226)[0m
[32m2022-05-24 16:21:03,442 - util.base_logger - INFO - Epoch=860 done. (train.py:226)[0m
[32m2022-05-24 16:21:12,600 - util.base_logger - INFO - Epoch=861 done. (train.py:226)[0m
[32m2022-05-24 16:21:21,712 - util.base_logger - INFO - Epoch=862 done. (train.py:226)[0m
[32m2022-05-24 16:21:31,041 - util.base_logger - INFO - Epoch=863 done. (train.py:226)[0m
[32m2022-05-24 16:21:40,136 - util.base_logger - INFO - Epoch=864 done. (train.py:226)[0m
[32m2022-05-24 16:21:49,285 - util.base_logger - INFO - Epoch=865 done. (train.py:226)[0m
[32m2022-05-24 16:21:58,411 - util.base_logger - INFO - Epoch=866 done. (train.py:226)[0m
[32m2022-05-24 16:22:07,761 - util.base_logger - INFO - Epoch=867 done. (train.py:226)[0m
[32m2022-05-24 16:22:16,872 - util.base_logger - INFO - Epoch=868 done. (train.py:226)[0m
[32m2022-05-24 16:22:25,982 - util.base_logger - INFO - Epoch=869 done. (train.py:226)[0m
[32m2022-05-24 16:22:35,223 - util.base_logger - INFO - Epoch=870 done. (train.py:226)[0m
[32m2022-05-24 16:22:44,367 - util.base_logger - INFO - Epoch=871 done. (train.py:226)[0m
[32m2022-05-24 16:22:53,468 - util.base_logger - INFO - Epoch=872 done. (train.py:226)[0m
[32m2022-05-24 16:23:02,802 - util.base_logger - INFO - Epoch=873 done. (train.py:226)[0m
[32m2022-05-24 16:23:11,909 - util.base_logger - INFO - Epoch=874 done. (train.py:226)[0m
[32m2022-05-24 16:23:21,039 - util.base_logger - INFO - Epoch=875 done. (train.py:226)[0m
[32m2022-05-24 16:23:30,310 - util.base_logger - INFO - Epoch=876 done. (train.py:226)[0m
[32m2022-05-24 16:23:39,415 - util.base_logger - INFO - Epoch=877 done. (train.py:226)[0m
[32m2022-05-24 16:23:48,540 - util.base_logger - INFO - Epoch=878 done. (train.py:226)[0m
[32m2022-05-24 16:23:57,698 - util.base_logger - INFO - Epoch=879 done. (train.py:226)[0m
[32m2022-05-24 16:24:07,005 - util.base_logger - INFO - Epoch=880 done. (train.py:226)[0m
[32m2022-05-24 16:24:16,119 - util.base_logger - INFO - Epoch=881 done. (train.py:226)[0m
[32m2022-05-24 16:24:25,210 - util.base_logger - INFO - Epoch=882 done. (train.py:226)[0m
[32m2022-05-24 16:24:34,477 - util.base_logger - INFO - Epoch=883 done. (train.py:226)[0m
[32m2022-05-24 16:24:43,599 - util.base_logger - INFO - Epoch=884 done. (train.py:226)[0m
[32m2022-05-24 16:24:52,738 - util.base_logger - INFO - Epoch=885 done. (train.py:226)[0m
[32m2022-05-24 16:25:02,059 - util.base_logger - INFO - Epoch=886 done. (train.py:226)[0m
[32m2022-05-24 16:25:11,187 - util.base_logger - INFO - Epoch=887 done. (train.py:226)[0m
[32m2022-05-24 16:25:20,283 - util.base_logger - INFO - Epoch=888 done. (train.py:226)[0m
[32m2022-05-24 16:25:29,571 - util.base_logger - INFO - Epoch=889 done. (train.py:226)[0m
[32m2022-05-24 16:25:38,699 - util.base_logger - INFO - Epoch=890 done. (train.py:226)[0m
[32m2022-05-24 16:25:47,839 - util.base_logger - INFO - Epoch=891 done. (train.py:226)[0m
[32m2022-05-24 16:25:56,983 - util.base_logger - INFO - Epoch=892 done. (train.py:226)[0m
[32m2022-05-24 16:26:06,290 - util.base_logger - INFO - Epoch=893 done. (train.py:226)[0m
[32m2022-05-24 16:26:15,389 - util.base_logger - INFO - Epoch=894 done. (train.py:226)[0m
[32m2022-05-24 16:26:24,528 - util.base_logger - INFO - Epoch=895 done. (train.py:226)[0m
[32m2022-05-24 16:26:33,789 - util.base_logger - INFO - Epoch=896 done. (train.py:226)[0m
[32m2022-05-24 16:26:42,924 - util.base_logger - INFO - Epoch=897 done. (train.py:226)[0m
[32m2022-05-24 16:26:52,054 - util.base_logger - INFO - Epoch=898 done. (train.py:226)[0m
[32m2022-05-24 16:27:01,362 - util.base_logger - INFO - Epoch=899 done. (train.py:226)[0m
[32m2022-05-24 16:27:10,475 - util.base_logger - INFO - Epoch=900 done. (train.py:226)[0m
[32m2022-05-24 16:27:19,600 - util.base_logger - INFO - Epoch=901 done. (train.py:226)[0m
[32m2022-05-24 16:27:28,873 - util.base_logger - INFO - Epoch=902 done. (train.py:226)[0m
[32m2022-05-24 16:27:38,012 - util.base_logger - INFO - Epoch=903 done. (train.py:226)[0m
[32m2022-05-24 16:27:47,145 - util.base_logger - INFO - Epoch=904 done. (train.py:226)[0m
[32m2022-05-24 16:27:56,279 - util.base_logger - INFO - Epoch=905 done. (train.py:226)[0m
[32m2022-05-24 16:28:05,546 - util.base_logger - INFO - Epoch=906 done. (train.py:226)[0m
[32m2022-05-24 16:28:14,675 - util.base_logger - INFO - Epoch=907 done. (train.py:226)[0m
[32m2022-05-24 16:28:23,780 - util.base_logger - INFO - Epoch=908 done. (train.py:226)[0m
[32m2022-05-24 16:28:33,112 - util.base_logger - INFO - Epoch=909 done. (train.py:226)[0m
[32m2022-05-24 16:28:42,245 - util.base_logger - INFO - Epoch=910 done. (train.py:226)[0m
[32m2022-05-24 16:28:51,375 - util.base_logger - INFO - Epoch=911 done. (train.py:226)[0m
[32m2022-05-24 16:29:00,636 - util.base_logger - INFO - Epoch=912 done. (train.py:226)[0m
[32m2022-05-24 16:29:09,761 - util.base_logger - INFO - Epoch=913 done. (train.py:226)[0m
[32m2022-05-24 16:29:18,878 - util.base_logger - INFO - Epoch=914 done. (train.py:226)[0m
[32m2022-05-24 16:29:28,200 - util.base_logger - INFO - Epoch=915 done. (train.py:226)[0m
[32m2022-05-24 16:29:37,342 - util.base_logger - INFO - Epoch=916 done. (train.py:226)[0m
[32m2022-05-24 16:29:46,486 - util.base_logger - INFO - Epoch=917 done. (train.py:226)[0m
[32m2022-05-24 16:29:55,645 - util.base_logger - INFO - Epoch=918 done. (train.py:226)[0m
[32m2022-05-24 16:30:04,927 - util.base_logger - INFO - Epoch=919 done. (train.py:226)[0m
[32m2022-05-24 16:30:14,036 - util.base_logger - INFO - Epoch=920 done. (train.py:226)[0m
[32m2022-05-24 16:30:23,147 - util.base_logger - INFO - Epoch=921 done. (train.py:226)[0m
[32m2022-05-24 16:30:32,476 - util.base_logger - INFO - Epoch=922 done. (train.py:226)[0m
[32m2022-05-24 16:30:41,614 - util.base_logger - INFO - Epoch=923 done. (train.py:226)[0m
[32m2022-05-24 16:30:50,736 - util.base_logger - INFO - Epoch=924 done. (train.py:226)[0m
[32m2022-05-24 16:30:59,986 - util.base_logger - INFO - Epoch=925 done. (train.py:226)[0m
[32m2022-05-24 16:31:09,110 - util.base_logger - INFO - Epoch=926 done. (train.py:226)[0m
[32m2022-05-24 16:31:18,240 - util.base_logger - INFO - Epoch=927 done. (train.py:226)[0m
[32m2022-05-24 16:31:27,565 - util.base_logger - INFO - Epoch=928 done. (train.py:226)[0m
[32m2022-05-24 16:31:36,650 - util.base_logger - INFO - Epoch=929 done. (train.py:226)[0m
[32m2022-05-24 16:31:45,780 - util.base_logger - INFO - Epoch=930 done. (train.py:226)[0m
[32m2022-05-24 16:31:54,897 - util.base_logger - INFO - Epoch=931 done. (train.py:226)[0m
[32m2022-05-24 16:32:04,165 - util.base_logger - INFO - Epoch=932 done. (train.py:226)[0m
[32m2022-05-24 16:32:13,296 - util.base_logger - INFO - Epoch=933 done. (train.py:226)[0m
[32m2022-05-24 16:32:22,448 - util.base_logger - INFO - Epoch=934 done. (train.py:226)[0m
[32m2022-05-24 16:32:31,750 - util.base_logger - INFO - Epoch=935 done. (train.py:226)[0m
[32m2022-05-24 16:32:40,874 - util.base_logger - INFO - Epoch=936 done. (train.py:226)[0m
[32m2022-05-24 16:32:50,023 - util.base_logger - INFO - Epoch=937 done. (train.py:226)[0m
[32m2022-05-24 16:32:59,301 - util.base_logger - INFO - Epoch=938 done. (train.py:226)[0m
[32m2022-05-24 16:33:08,447 - util.base_logger - INFO - Epoch=939 done. (train.py:226)[0m
[32m2022-05-24 16:33:17,611 - util.base_logger - INFO - Epoch=940 done. (train.py:226)[0m
[32m2022-05-24 16:33:26,940 - util.base_logger - INFO - Epoch=941 done. (train.py:226)[0m
[32m2022-05-24 16:33:36,056 - util.base_logger - INFO - Epoch=942 done. (train.py:226)[0m
[32m2022-05-24 16:33:45,178 - util.base_logger - INFO - Epoch=943 done. (train.py:226)[0m
[32m2022-05-24 16:33:54,307 - util.base_logger - INFO - Epoch=944 done. (train.py:226)[0m
[32m2022-05-24 16:34:03,599 - util.base_logger - INFO - Epoch=945 done. (train.py:226)[0m
[32m2022-05-24 16:34:12,747 - util.base_logger - INFO - Epoch=946 done. (train.py:226)[0m
[32m2022-05-24 16:34:21,877 - util.base_logger - INFO - Epoch=947 done. (train.py:226)[0m
[32m2022-05-24 16:34:31,135 - util.base_logger - INFO - Epoch=948 done. (train.py:226)[0m
[32m2022-05-24 16:34:40,248 - util.base_logger - INFO - Epoch=949 done. (train.py:226)[0m
[32m2022-05-24 16:34:49,380 - util.base_logger - INFO - Epoch=950 done. (train.py:226)[0m
[32m2022-05-24 16:34:58,704 - util.base_logger - INFO - Epoch=951 done. (train.py:226)[0m
[32m2022-05-24 16:35:07,847 - util.base_logger - INFO - Epoch=952 done. (train.py:226)[0m
[32m2022-05-24 16:35:16,989 - util.base_logger - INFO - Epoch=953 done. (train.py:226)[0m
[32m2022-05-24 16:35:26,260 - util.base_logger - INFO - Epoch=954 done. (train.py:226)[0m
[32m2022-05-24 16:35:35,382 - util.base_logger - INFO - Epoch=955 done. (train.py:226)[0m
[32m2022-05-24 16:35:44,505 - util.base_logger - INFO - Epoch=956 done. (train.py:226)[0m
[32m2022-05-24 16:35:53,651 - util.base_logger - INFO - Epoch=957 done. (train.py:226)[0m
[32m2022-05-24 16:36:03,007 - util.base_logger - INFO - Epoch=958 done. (train.py:226)[0m
[32m2022-05-24 16:36:12,138 - util.base_logger - INFO - Epoch=959 done. (train.py:226)[0m
[32m2022-05-24 16:36:21,251 - util.base_logger - INFO - Epoch=960 done. (train.py:226)[0m
[32m2022-05-24 16:36:30,540 - util.base_logger - INFO - Epoch=961 done. (train.py:226)[0m
[32m2022-05-24 16:36:39,654 - util.base_logger - INFO - Epoch=962 done. (train.py:226)[0m
[32m2022-05-24 16:36:48,799 - util.base_logger - INFO - Epoch=963 done. (train.py:226)[0m
[32m2022-05-24 16:36:58,132 - util.base_logger - INFO - Epoch=964 done. (train.py:226)[0m
[32m2022-05-24 16:37:07,243 - util.base_logger - INFO - Epoch=965 done. (train.py:226)[0m
[32m2022-05-24 16:37:16,349 - util.base_logger - INFO - Epoch=966 done. (train.py:226)[0m
[32m2022-05-24 16:37:25,613 - util.base_logger - INFO - Epoch=967 done. (train.py:226)[0m
[32m2022-05-24 16:37:34,727 - util.base_logger - INFO - Epoch=968 done. (train.py:226)[0m
[32m2022-05-24 16:37:43,860 - util.base_logger - INFO - Epoch=969 done. (train.py:226)[0m
[32m2022-05-24 16:37:53,032 - util.base_logger - INFO - Epoch=970 done. (train.py:226)[0m
[32m2022-05-24 16:38:02,333 - util.base_logger - INFO - Epoch=971 done. (train.py:226)[0m
[32m2022-05-24 16:38:11,446 - util.base_logger - INFO - Epoch=972 done. (train.py:226)[0m
[32m2022-05-24 16:38:20,550 - util.base_logger - INFO - Epoch=973 done. (train.py:226)[0m
[32m2022-05-24 16:38:29,853 - util.base_logger - INFO - Epoch=974 done. (train.py:226)[0m
[32m2022-05-24 16:38:38,971 - util.base_logger - INFO - Epoch=975 done. (train.py:226)[0m
[32m2022-05-24 16:38:48,127 - util.base_logger - INFO - Epoch=976 done. (train.py:226)[0m
[32m2022-05-24 16:38:57,428 - util.base_logger - INFO - Epoch=977 done. (train.py:226)[0m
[32m2022-05-24 16:39:06,561 - util.base_logger - INFO - Epoch=978 done. (train.py:226)[0m
[32m2022-05-24 16:39:15,685 - util.base_logger - INFO - Epoch=979 done. (train.py:226)[0m
[32m2022-05-24 16:39:24,915 - util.base_logger - INFO - Epoch=980 done. (train.py:226)[0m
[32m2022-05-24 16:39:34,015 - util.base_logger - INFO - Epoch=981 done. (train.py:226)[0m
[32m2022-05-24 16:39:43,143 - util.base_logger - INFO - Epoch=982 done. (train.py:226)[0m
[32m2022-05-24 16:39:52,233 - util.base_logger - INFO - Epoch=983 done. (train.py:226)[0m
[32m2022-05-24 16:40:01,485 - util.base_logger - INFO - Epoch=984 done. (train.py:226)[0m
[32m2022-05-24 16:40:10,562 - util.base_logger - INFO - Epoch=985 done. (train.py:226)[0m
[32m2022-05-24 16:40:19,649 - util.base_logger - INFO - Epoch=986 done. (train.py:226)[0m
[32m2022-05-24 16:40:28,915 - util.base_logger - INFO - Epoch=987 done. (train.py:226)[0m
[32m2022-05-24 16:40:38,022 - util.base_logger - INFO - Epoch=988 done. (train.py:226)[0m
[32m2022-05-24 16:40:47,102 - util.base_logger - INFO - Epoch=989 done. (train.py:226)[0m
[32m2022-05-24 16:40:56,323 - util.base_logger - INFO - Epoch=990 done. (train.py:226)[0m
[32m2022-05-24 16:41:05,394 - util.base_logger - INFO - Epoch=991 done. (train.py:226)[0m
[32m2022-05-24 16:41:14,462 - util.base_logger - INFO - Epoch=992 done. (train.py:226)[0m
[32m2022-05-24 16:41:23,709 - util.base_logger - INFO - Epoch=993 done. (train.py:226)[0m
[32m2022-05-24 16:41:32,814 - util.base_logger - INFO - Epoch=994 done. (train.py:226)[0m
[32m2022-05-24 16:41:41,903 - util.base_logger - INFO - Epoch=995 done. (train.py:226)[0m
[32m2022-05-24 16:41:50,995 - util.base_logger - INFO - Epoch=996 done. (train.py:226)[0m
[32m2022-05-24 16:42:00,220 - util.base_logger - INFO - Epoch=997 done. (train.py:226)[0m
[32m2022-05-24 16:42:09,328 - util.base_logger - INFO - Epoch=998 done. (train.py:226)[0m
[32m2022-05-24 16:42:18,413 - util.base_logger - INFO - Epoch=999 done. (train.py:226)[0m
[33;20m2022-05-24 16:42:20,627 - util.base_logger - WARNING - ./. already exists (utils.py:27)[0m
[33;20m2022-05-24 16:42:20,628 - util.base_logger - WARNING - ././out already exists (utils.py:27)[0m
[33;20m2022-05-24 16:42:20,628 - util.base_logger - WARNING - ././out/cluster_config_1000_gpu-2022-05-24-14-07-45 already exists (utils.py:27)[0m
[32m2022-05-24 16:42:20,630 - util.base_logger - INFO - Created folder ././out/cluster_config_1000_gpu-2022-05-24-14-07-45/net_eval (utils.py:30)[0m
[33;20m2022-05-24 16:42:20,868 - util.base_logger - WARNING - ./. already exists (utils.py:27)[0m
[33;20m2022-05-24 16:42:20,868 - util.base_logger - WARNING - ././out already exists (utils.py:27)[0m
[33;20m2022-05-24 16:42:20,868 - util.base_logger - WARNING - ././out/cluster_config_1000_gpu-2022-05-24-14-07-45 already exists (utils.py:27)[0m
[33;20m2022-05-24 16:42:20,868 - util.base_logger - WARNING - ././out/cluster_config_1000_gpu-2022-05-24-14-07-45/net_eval already exists (utils.py:27)[0m
[33;20m2022-05-24 16:42:21,068 - util.base_logger - WARNING - ./. already exists (utils.py:27)[0m
[33;20m2022-05-24 16:42:21,068 - util.base_logger - WARNING - ././out already exists (utils.py:27)[0m
[33;20m2022-05-24 16:42:21,068 - util.base_logger - WARNING - ././out/cluster_config_1000_gpu-2022-05-24-14-07-45 already exists (utils.py:27)[0m
[33;20m2022-05-24 16:42:21,069 - util.base_logger - WARNING - ././out/cluster_config_1000_gpu-2022-05-24-14-07-45/net_eval already exists (utils.py:27)[0m
[33;20m2022-05-24 16:42:21,267 - util.base_logger - WARNING - ./. already exists (utils.py:27)[0m
[33;20m2022-05-24 16:42:21,267 - util.base_logger - WARNING - ././out already exists (utils.py:27)[0m
[33;20m2022-05-24 16:42:21,267 - util.base_logger - WARNING - ././out/cluster_config_1000_gpu-2022-05-24-14-07-45 already exists (utils.py:27)[0m
[33;20m2022-05-24 16:42:21,267 - util.base_logger - WARNING - ././out/cluster_config_1000_gpu-2022-05-24-14-07-45/net_eval already exists (utils.py:27)[0m
[32m2022-05-24 16:42:21,873 - util.base_logger - INFO - {
    "root": "out/cluster_config_1000_gpu-2022-05-24-14-07-45",
    "name": "results-cluster_config_1000_gpu",
    "model": "autoencoders.resnet_ae",
    "epochs": 1000,
    "batch_size": 512,
    "optimizer": "torch.optim.adam",
    "optimizer_args": {
        "lr": 0.0001,
        "betas": [
            0.9,
            0.999
        ],
        "eps": 1e-08,
        "weight_decay": 0,
        "amsgrad": false
    },
    "loss": "MSELoss()",
    "loss_args": null,
    "train_loss": [
        7.380398783011055e-05,
        2.77379591583066e-05,
        2.36699342964387e-05,
        2.2577580641161685e-05,
        2.160025953002139e-05,
        2.039736169527205e-05,
        2.00562915452737e-05,
        1.9664912824094104e-05,
        1.9334806172544067e-05,
        1.9291242586942927e-05,
        1.8564587607969302e-05,
        1.788758732348106e-05,
        1.7292193966828266e-05,
        1.686485333670112e-05,
        1.6566073323216233e-05,
        1.6296545846727384e-05,
        1.61971950176336e-05,
        1.61892180230888e-05,
        1.6218084833063222e-05,
        1.6373778321136993e-05,
        1.5577535097456237e-05,
        1.469421799208366e-05,
        1.418144478409347e-05,
        1.3885740570448529e-05,
        1.3708642964249637e-05,
        1.3656287329551891e-05,
        1.3829254540426894e-05,
        1.3785220514503724e-05,
        1.399255749563552e-05,
        1.4387087761082094e-05,
        1.3921911251200548e-05,
        1.3055231798747646e-05,
        1.2493288149968425e-05,
        1.2259866411732368e-05,
        1.1934129827370815e-05,
        1.1513362006641348e-05,
        1.1214865396462058e-05,
        1.096008562832614e-05,
        1.0901638164615625e-05,
        1.0953168204698686e-05,
        1.0942671807794722e-05,
        1.0913362446111847e-05,
        1.0643146199427056e-05,
        1.045798904576852e-05,
        1.0123377139266996e-05,
        9.922849104221104e-06,
        9.810939626541717e-06,
        9.731865259777488e-06,
        9.589747634556463e-06,
        9.508375599550896e-06,
        9.491782856464351e-06,
        9.314973563538244e-06,
        9.288059021980638e-06,
        9.256493104848282e-06,
        9.303462832881215e-06,
        9.236887933231517e-06,
        9.141818791916983e-06,
        8.919932661504997e-06,
        8.759315000079979e-06,
        8.624395975940223e-06,
        8.648558120829435e-06,
        8.719067999109275e-06,
        8.756340149165237e-06,
        8.725717993191018e-06,
        8.785270068584473e-06,
        8.66244566839874e-06,
        8.734589387357243e-06,
        8.648519299355601e-06,
        8.53213065501701e-06,
        8.403732754687093e-06,
        8.202683739390335e-06,
        8.113102239260572e-06,
        8.040347458520017e-06,
        7.927683842614971e-06,
        7.929467339093266e-06,
        7.966478475293967e-06,
        8.043443683823436e-06,
        7.99388506102919e-06,
        7.943404706462883e-06,
        7.836872788504109e-06,
        7.750307743077656e-06,
        7.694524689423675e-06,
        7.731644663230187e-06,
        7.72786991319567e-06,
        7.801368292099007e-06,
        7.826608272983473e-06,
        7.790468852355831e-06,
        7.81173856831269e-06,
        7.817496061293649e-06,
        7.814790931242638e-06,
        7.870491235583086e-06,
        8.032659464643406e-06,
        7.682452094855732e-06,
        7.575656904456787e-06,
        7.519650774627282e-06,
        7.567887961587942e-06,
        7.616763313349484e-06,
        7.924598844769834e-06,
        8.223674765610082e-06,
        7.613548234107206e-06,
        7.64400394219404e-06,
        7.761807344475745e-06,
        8.288400735285537e-06,
        7.964164047697988e-06,
        7.602119270770272e-06,
        7.128522363574499e-06,
        6.78518776944606e-06,
        6.448496164512855e-06,
        6.180104527098208e-06,
        6.0324892770924784e-06,
        5.944907344730027e-06,
        5.911264773345641e-06,
        5.924992779714723e-06,
        5.941725915129486e-06,
        5.960113840416214e-06,
        6.023424790283639e-06,
        6.182074373197525e-06,
        6.340472009328871e-06,
        6.4874686034699795e-06,
        6.646833208537665e-06,
        6.594778751226714e-06,
        6.6815794423446715e-06,
        6.8604795629614e-06,
        7.006724735725351e-06,
        7.081859901287663e-06,
        6.649864360409397e-06,
        6.5252610463814015e-06,
        6.472552745107708e-06,
        6.584301895992121e-06,
        6.618053632313468e-06,
        6.469707386394048e-06,
        6.221666384546479e-06,
        6.053461023507768e-06,
        5.8631918411536945e-06,
        5.7438521428746255e-06,
        5.702480464821774e-06,
        5.72821206435337e-06,
        5.785901690996281e-06,
        5.936546456218018e-06,
        6.11181748915989e-06,
        6.2165335703232135e-06,
        6.358649886219658e-06,
        6.362810166877113e-06,
        6.466967984772124e-06,
        6.5854332506296316e-06,
        6.5869986136327e-06,
        6.494917318279374e-06,
        6.387539052911302e-06,
        6.1585018499003476e-06,
        5.943515467233971e-06,
        5.897739974151562e-06,
        5.673071365338932e-06,
        5.745612627973406e-06,
        6.009693674268676e-06,
        6.3082947358164325e-06,
        6.802760083718037e-06,
        6.779610537726819e-06,
        6.192601670161608e-06,
        5.75633530889812e-06,
        5.494554582226466e-06,
        5.585281217634955e-06,
        5.789475230575781e-06,
        5.969003597859691e-06,
        6.0658100819610335e-06,
        6.115418753687396e-06,
        6.069651673015422e-06,
        6.021273406331136e-06,
        6.019633182695138e-06,
        6.015957875862565e-06,
        5.945052745224771e-06,
        5.998389751429248e-06,
        6.2436100098652865e-06,
        6.137649906684237e-06,
        6.161011432325155e-06,
        6.158639787244979e-06,
        5.859380790095163e-06,
        5.743846283647124e-06,
        5.5719030627939725e-06,
        5.4623383890408494e-06,
        5.360953131412782e-06,
        5.371650444173736e-06,
        5.5002956759170455e-06,
        5.6282972764293464e-06,
        5.65230436455503e-06,
        5.582368003174874e-06,
        5.4930665348399195e-06,
        5.514425841331247e-06,
        5.5412073555114005e-06,
        5.420622293156911e-06,
        5.216596692090343e-06,
        5.033382183312952e-06,
        4.906684244553047e-06,
        4.841655395368844e-06,
        4.777113222131477e-06,
        4.766621178731744e-06,
        4.768608275182419e-06,
        4.812328685135273e-06,
        4.840168002644588e-06,
        4.818690889674065e-06,
        4.810044241072206e-06,
        4.755249563811467e-06,
        4.784778073696116e-06,
        4.660981663674405e-06,
        4.629726285599596e-06,
        4.552996231764912e-06,
        4.509501582102971e-06,
        4.528416477800522e-06,
        4.608445309451339e-06,
        4.749637667723938e-06,
        4.83603868751348e-06,
        4.905991808248274e-06,
        4.840462437009789e-06,
        4.78617296263871e-06,
        4.786036596483576e-06,
        4.812962529165037e-06,
        4.8174767202567944e-06,
        4.895203071898438e-06,
        4.8517043960634105e-06,
        4.81035055755798e-06,
        4.762205383382266e-06,
        4.7208144909911865e-06,
        4.830757395882706e-06,
        4.628362656781379e-06,
        4.486620807715161e-06,
        4.443767661631787e-06,
        4.463566842418371e-06,
        4.526497924491673e-06,
        4.667461641959277e-06,
        4.790286434942385e-06,
        4.931402655990587e-06,
        4.9056686014754026e-06,
        4.821711501482919e-06,
        4.79381499922252e-06,
        4.724376410315028e-06,
        4.688037121290064e-06,
        4.661418618020271e-06,
        4.661822479187345e-06,
        4.6308145961914995e-06,
        4.632689516258679e-06,
        4.650753285512213e-06,
        4.7289817303976375e-06,
        4.810145353662989e-06,
        4.759795833355462e-06,
        4.696453197833172e-06,
        4.6082858009842325e-06,
        4.592590436232537e-06,
        4.654433764176883e-06,
        4.78502239366297e-06,
        5.006441946260439e-06,
        5.249634879248046e-06,
        5.461732859155153e-06,
        5.667052400239104e-06,
        5.731077423000008e-06,
        5.841126317716377e-06,
        5.821047628864814e-06,
        5.7361174715766935e-06,
        5.557360067339315e-06,
        5.197622647654837e-06,
        5.029098924344206e-06,
        4.691236815968512e-06,
        4.364979382288551e-06,
        4.226332187336256e-06,
        4.187535459737929e-06,
        4.142312256130433e-06,
        4.149120432988143e-06,
        4.148922332179005e-06,
        4.1493637545950034e-06,
        4.125214359253901e-06,
        4.265792022368735e-06,
        4.311627089719589e-06,
        4.279972187615381e-06,
        4.305303182924797e-06,
        4.378249976115188e-06,
        4.343759027868405e-06,
        4.31462910911939e-06,
        4.280471874975243e-06,
        4.273063487362949e-06,
        4.28494141809878e-06,
        4.2035237858646706e-06,
        4.146574304040961e-06,
        4.0799535943974794e-06,
        4.024533649182983e-06,
        3.944426290790407e-06,
        3.965701931440996e-06,
        3.969905321610287e-06,
        4.085453346834627e-06,
        4.132696003576513e-06,
        4.271180023952876e-06,
        4.317368445275085e-06,
        4.391740896535727e-06,
        4.402027474175554e-06,
        4.391510291743862e-06,
        4.336334371898188e-06,
        4.17164227240857e-06,
        4.024679606140674e-06,
        3.920056978019333e-06,
        3.857919772172735e-06,
        3.866776240038734e-06,
        3.904152039501858e-06,
        3.910028697386265e-06,
        3.913815575039715e-06,
        3.940257204925144e-06,
        3.901469020669711e-06,
        3.872907266956157e-06,
        3.829393255656971e-06,
        3.8066939536561184e-06,
        3.817065162763563e-06,
        3.728826309525604e-06,
        3.725954682487535e-06,
        3.7686841464976576e-06,
        3.7637883691232652e-06,
        3.7317290002893638e-06,
        3.7191907771672274e-06,
        3.6888402733104935e-06,
        3.6727769392518056e-06,
        3.6432991984277925e-06,
        3.6087940112761887e-06,
        3.598422278438911e-06,
        3.5775389768990823e-06,
        3.5967228405987274e-06,
        3.5990763515334553e-06,
        3.628081802614468e-06,
        3.6844699606236546e-06,
        3.712509408395208e-06,
        3.7255000358932665e-06,
        3.6546632192654983e-06,
        3.681951491158252e-06,
        3.7691514935403638e-06,
        3.8990137588484665e-06,
        3.92374166288922e-06,
        3.9618764287124e-06,
        3.946324647767591e-06,
        3.860919974884679e-06,
        3.7484171113050906e-06,
        3.667780573016461e-06,
        3.6064996490794182e-06,
        3.5803508332700196e-06,
        3.546804268110549e-06,
        3.5745023422306658e-06,
        3.6314923967498357e-06,
        3.650090059467916e-06,
        3.666272001967104e-06,
        3.6560332146744917e-06,
        3.6672682179412766e-06,
        3.737946753593733e-06,
        3.875289206600853e-06,
        3.851316275480183e-06,
        3.8040986432043556e-06,
        3.801127360199099e-06,
        3.723593626569909e-06,
        3.7181133830690205e-06,
        3.7630616121479065e-06,
        3.7674662782384442e-06,
        3.7970804343172625e-06,
        3.7971473244368053e-06,
        3.7582615136672846e-06,
        3.6794931033245705e-06,
        3.683117722929295e-06,
        3.675539123121443e-06,
        3.6473794302536636e-06,
        3.5877691226494593e-06,
        3.6112207298215626e-06,
        3.69644361935293e-06,
        3.7696659108017644e-06,
        3.720916123267561e-06,
        3.7067496404627884e-06,
        3.718212580772605e-06,
        3.6738866409340536e-06,
        3.6219075843524224e-06,
        3.5765931371881728e-06,
        3.62562608252962e-06,
        3.559819494544178e-06,
        3.537907914944495e-06,
        3.544185013385508e-06,
        3.603448939171753e-06,
        3.6270273871626613e-06,
        3.71547491400575e-06,
        3.6859476316128955e-06,
        3.6110401903283707e-06,
        3.5173398823972646e-06,
        3.4033050599579874e-06,
        3.297415431172522e-06,
        3.194533075451742e-06,
        3.1197304946996143e-06,
        3.087279326851343e-06,
        3.089863344378592e-06,
        3.1637926874678383e-06,
        3.2764911641398596e-06,
        3.30401105555103e-06,
        3.584474501938873e-06,
        3.4934210754814702e-06,
        3.4631482309389595e-06,
        3.398509675045858e-06,
        3.499983835812989e-06,
        3.4784236625643464e-06,
        3.3925528046642943e-06,
        3.3303060574499206e-06,
        3.2102042995314715e-06,
        3.2221850613465997e-06,
        3.1642367939992275e-06,
        3.129409660299661e-06,
        3.0352550331054313e-06,
        3.044997389983078e-06,
        3.0033482169517983e-06,
        3.0046309295108967e-06,
        2.9905102730664666e-06,
        3.0079036354013053e-06,
        3.04934338196582e-06,
        3.050596634721842e-06,
        3.068523726855654e-06,
        3.0462033270220458e-06,
        3.0347782752922977e-06,
        3.029317638927014e-06,
        3.055271447206598e-06,
        3.0897085494799787e-06,
        3.135707724300531e-06,
        3.2201954117129913e-06,
        3.3676094676302606e-06,
        3.4476902794669524e-06,
        3.275658597371782e-06,
        3.1711094223594474e-06,
        3.1780355693619237e-06,
        3.2454016447563753e-06,
        3.317404660422072e-06,
        3.4504978968995577e-06,
        3.4858904951526344e-06,
        3.4951109716187676e-06,
        3.4499571458475177e-06,
        3.3924261111445054e-06,
        3.3207851400920716e-06,
        3.271619511070876e-06,
        3.268269129499686e-06,
        3.2079989550396654e-06,
        3.1666196501711305e-06,
        3.1932117541840546e-06,
        3.227558594894119e-06,
        3.2181501157846815e-06,
        3.225180615956281e-06,
        3.224139702914218e-06,
        3.2256661789772226e-06,
        3.198876890682368e-06,
        3.1351690681678206e-06,
        3.1022821898320362e-06,
        3.0923259547840488e-06,
        3.1669281925086936e-06,
        3.1687329000452055e-06,
        3.1447361392171577e-06,
        3.0938280283101716e-06,
        3.0778093258531454e-06,
        3.053911680895895e-06,
        3.0657972875470837e-06,
        3.0625782312313895e-06,
        2.97572465976691e-06,
        2.9663592901693088e-06,
        2.9598511286728254e-06,
        3.0054425307191174e-06,
        2.983570754586703e-06,
        3.015767373369831e-06,
        3.062839539684685e-06,
        3.057777036191432e-06,
        3.033437101390649e-06,
        2.993155714283131e-06,
        2.949272686217289e-06,
        2.9139152269626607e-06,
        2.887726951383696e-06,
        2.879635898301316e-06,
        2.8803481217738228e-06,
        2.8864297871545337e-06,
        2.933872132261611e-06,
        3.0155825294720773e-06,
        3.077494089593662e-06,
        3.1452019314369236e-06,
        3.194745529731601e-06,
        3.1862223340711914e-06,
        3.1980808049704344e-06,
        3.2355617595639866e-06,
        3.227140740320583e-06,
        3.1786580386343898e-06,
        3.1413902748157742e-06,
        3.1325454272054324e-06,
        3.138518811443277e-06,
        3.097079654074823e-06,
        3.040208240729268e-06,
        3.019397868568613e-06,
        3.0223807063632985e-06,
        2.9794021597181574e-06,
        2.8574425520517564e-06,
        2.7629784823593145e-06,
        2.6776341693395973e-06,
        2.620954899376335e-06,
        2.5372468656523137e-06,
        2.467112092498434e-06,
        2.425212412941654e-06,
        2.387975352784467e-06,
        2.3651983751028327e-06,
        2.3617912834107197e-06,
        2.397510934577626e-06,
        2.4306657170892835e-06,
        2.4469498196389206e-06,
        2.4643344094990668e-06,
        2.5121543310822506e-06,
        2.5857214004264326e-06,
        2.6841262752434025e-06,
        2.7286278446079653e-06,
        2.7502198835436035e-06,
        2.756008014719764e-06,
        2.770868423186022e-06,
        2.7822840641453414e-06,
        2.790280796758203e-06,
        2.8130267434472628e-06,
        2.8511803962775265e-06,
        2.8323719650349855e-06,
        2.8239196202009774e-06,
        2.8085957926657565e-06,
        2.8648733454809466e-06,
        2.9074028756275183e-06,
        2.943520446901212e-06,
        2.897716148678016e-06,
        2.8412952575542986e-06,
        2.831497418047523e-06,
        2.840281234765822e-06,
        2.8874320423883363e-06,
        2.943512394555038e-06,
        2.9690812614077458e-06,
        2.984486594398146e-06,
        2.950595938738734e-06,
        3.031603212282533e-06,
        3.0543765402218968e-06,
        2.8466363362187616e-06,
        2.8213100708443943e-06,
        2.8366484155159075e-06,
        2.8551101539759013e-06,
        2.988379118178743e-06,
        2.967314704316218e-06,
        2.887771779384046e-06,
        2.8394210576156387e-06,
        2.83832822985393e-06,
        3.0292655769083522e-06,
        2.973774257137623e-06,
        2.9529822809873347e-06,
        2.9244520983627433e-06,
        2.9308638444703776e-06,
        2.958921279088365e-06,
        2.895556532347209e-06,
        2.83687946220482e-06,
        2.7792215703164457e-06,
        2.7500202770111987e-06,
        2.7199625708639378e-06,
        2.658513496917552e-06,
        2.6204640826905102e-06,
        2.598788737978393e-06,
        2.636555910925397e-06,
        2.628175546210241e-06,
        2.5875547658047157e-06,
        2.5490644037918373e-06,
        2.5194615242709743e-06,
        2.445108156416104e-06,
        2.3991808798816173e-06,
        2.3871455519645674e-06,
        2.416117926233348e-06,
        2.457220603550987e-06,
        2.4698809375568776e-06,
        2.4524530745193236e-06,
        2.4609794289252854e-06,
        2.478358290490388e-06,
        2.5670840488069836e-06,
        2.6927967862857557e-06,
        2.7614499277435373e-06,
        2.839161942281017e-06,
        2.875791917729567e-06,
        2.897171715150591e-06,
        2.8775420100644856e-06,
        2.7421297142553156e-06,
        2.7008113911791504e-06,
        2.5864946220578662e-06,
        2.512948992903702e-06,
        2.4301192541087578e-06,
        2.3464442971969696e-06,
        2.3175788620133596e-06,
        2.31762655416123e-06,
        2.3148172346066695e-06,
        2.310657805010193e-06,
        2.324046499914054e-06,
        2.307425589982457e-06,
        2.2871080966938418e-06,
        2.2731208931573514e-06,
        2.298380399344634e-06,
        2.3591048603346626e-06,
        2.4129943524127383e-06,
        2.4809697711007277e-06,
        2.5575570430898317e-06,
        2.5330051122727934e-06,
        2.5280025758453604e-06,
        2.501634153570578e-06,
        2.4858884125556203e-06,
        2.4584248712011133e-06,
        2.3811894835443095e-06,
        2.3539215061825643e-06,
        2.3293556310587356e-06,
        2.3520297940276298e-06,
        2.390703837912695e-06,
        2.3885764800662584e-06,
        2.3637232900296406e-06,
        2.3624457820357523e-06,
        2.376122675646496e-06,
        2.402811407813514e-06,
        2.4472735828747398e-06,
        2.5171925793376367e-06,
        2.5148365479530905e-06,
        2.4880885688487727e-06,
        2.504969003278923e-06,
        2.5280728865753714e-06,
        2.542365539170101e-06,
        2.5496503101753717e-06,
        2.584384039932626e-06,
        2.6066658146910898e-06,
        2.616825731544241e-06,
        2.5588223252654427e-06,
        2.5520925114839185e-06,
        2.5728358298594585e-06,
        2.540384334680041e-06,
        2.543651148609835e-06,
        2.5240823434833133e-06,
        2.4931389938227644e-06,
        2.462909962554052e-06,
        2.420224753714774e-06,
        2.3962087294488252e-06,
        2.3613235108375247e-06,
        2.3524806926802845e-06,
        2.3669377800758573e-06,
        2.441306679793573e-06,
        2.5368381927174035e-06,
        2.6034451544527836e-06,
        2.6105186823034327e-06,
        2.6183048918901927e-06,
        2.6524712094736224e-06,
        2.628739325008353e-06,
        2.609934756273327e-06,
        2.615034706449602e-06,
        2.6571009648262273e-06,
        2.695548151861041e-06,
        2.6177811456911508e-06,
        2.5735913592424885e-06,
        2.4502455860085163e-06,
        2.359585742520227e-06,
        2.3187697581862006e-06,
        2.3291014420578505e-06,
        2.3249962021324782e-06,
        2.287169078486212e-06,
        2.2810686079312037e-06,
        2.2693180417440097e-06,
        2.2485586350426785e-06,
        2.2082700793830784e-06,
        2.1447901939115626e-06,
        2.0996040851061066e-06,
        2.0755902621421102e-06,
        2.076527640342914e-06,
        2.0680275215282484e-06,
        2.050132753345034e-06,
        2.0342256462587207e-06,
        2.017297446031192e-06,
        2.011876924097764e-06,
        2.0283230391977694e-06,
        2.0599547989900044e-06,
        2.0829206467424787e-06,
        2.1069275875691455e-06,
        2.151785538718157e-06,
        2.18399845859232e-06,
        2.200203592503153e-06,
        2.241494943492945e-06,
        2.263479910735405e-06,
        2.294022214276729e-06,
        2.3073885688299235e-06,
        2.3379085811202526e-06,
        2.302654165710165e-06,
        2.278616128424694e-06,
        2.287911400057529e-06,
        2.3534440446074678e-06,
        2.4447020857637987e-06,
        2.466489672325641e-06,
        2.4615537641528334e-06,
        2.4533282270694053e-06,
        2.4308235070678765e-06,
        2.4119526374093697e-06,
        2.3715203179107527e-06,
        2.381022168229388e-06,
        2.297213365612286e-06,
        2.26353338027799e-06,
        2.185507209673807e-06,
        2.134506374036635e-06,
        2.1041159603300097e-06,
        2.142538270197781e-06,
        2.1303028774847217e-06,
        2.21069789448779e-06,
        2.2999333237641792e-06,
        2.2625662804091934e-06,
        2.207397267250686e-06,
        2.1016945772823045e-06,
        2.038605549748116e-06,
        2.0019518264249734e-06,
        1.991163998419066e-06,
        1.992215705205645e-06,
        2.0084256263344778e-06,
        2.013007804105114e-06,
        1.9847361639856404e-06,
        1.9723819664333937e-06,
        1.959370013311218e-06,
        1.926033005550971e-06,
        1.91366273603949e-06,
        1.9253365594396675e-06,
        1.93199499048168e-06,
        1.9370858081190016e-06,
        1.9585347133280775e-06,
        1.978992623494749e-06,
        1.9927400978836985e-06,
        2.0142004269497456e-06,
        2.0429773599749437e-06,
        2.069778505840538e-06,
        2.1287166634876983e-06,
        2.1807877655873885e-06,
        2.210628123854168e-06,
        2.200907845462272e-06,
        2.1937845632775364e-06,
        2.184835460697416e-06,
        2.188444990336342e-06,
        2.188836232887765e-06,
        2.18137877833679e-06,
        2.145142925953741e-06,
        2.150607539392441e-06,
        2.165925065803114e-06,
        2.0733841975217847e-06,
        1.974902825416157e-06,
        1.8943917777053582e-06,
        1.8562463245202843e-06,
        1.8476640046717547e-06,
        1.8530340439592983e-06,
        1.8492525099867949e-06,
        1.8783941062744176e-06,
        1.9260093967921156e-06,
        1.9674000191350003e-06,
        2.0062747251954146e-06,
        2.0337528982521168e-06,
        2.059333639042121e-06,
        2.06217085539354e-06,
        2.0545204391324087e-06,
        2.0571504548688517e-06,
        2.0806737148286625e-06,
        2.1237914263102287e-06,
        2.1265044286752846e-06,
        2.1019976941061335e-06,
        2.0515936649796345e-06,
        2.030282770764697e-06,
        2.05225772167411e-06,
        2.084105151774707e-06,
        2.063794033260146e-06,
        2.037116888615672e-06,
        2.0009779835345835e-06,
        1.9844068934033015e-06,
        1.972493005341159e-06,
        1.991952162717283e-06,
        2.0080972231796744e-06,
        2.0472634994577676e-06,
        2.0360297809657286e-06,
        2.008462352888981e-06,
        2.006068285625351e-06,
        1.997757887942503e-06,
        1.951859334716017e-06,
        1.9257533665535384e-06,
        1.8994298459286138e-06,
        1.8808972893753341e-06,
        1.8832806038108408e-06,
        1.8377987922319969e-06,
        1.7709760701721913e-06,
        1.7641583761614317e-06,
        1.75509025597688e-06,
        1.752073400292918e-06,
        1.7665292765632528e-06,
        1.7901588700456148e-06,
        1.817889971878292e-06,
        1.882323430259025e-06,
        1.9255759367062275e-06,
        1.954174850688053e-06,
        1.9636253836663837e-06,
        1.977052073532936e-06,
        1.9704896650823982e-06,
        1.9450907485603315e-06,
        1.9423963713374434e-06,
        1.946508182435557e-06,
        1.9360350260429077e-06,
        1.9280578678325355e-06,
        1.9442720524495356e-06,
        1.9619348737831887e-06,
        1.9696726138106076e-06,
        2.022109753963533e-06,
        2.03654377269664e-06,
        2.103061520328367e-06,
        2.1548766412891514e-06,
        2.2043677520346796e-06,
        2.157017436079104e-06,
        2.124406522448646e-06,
        2.101804936977943e-06,
        2.084606533073246e-06,
        2.0251165851311696e-06,
        1.9615131567021383e-06,
        1.9109233917005158e-06,
        1.8714536385343333e-06,
        1.8293498189087851e-06,
        1.7848815964046499e-06,
        1.7422507245354925e-06,
        1.6971052457084474e-06,
        1.6628718334235095e-06,
        1.6423777367987248e-06,
        1.6498765496399084e-06,
        1.6676787976749031e-06,
        1.7047988287643652e-06,
        1.7537360788454831e-06,
        1.8289361705404662e-06,
        1.8729431016280835e-06,
        1.8747147896353132e-06,
        1.9042379476557376e-06,
        1.953110738051067e-06,
        2.031349624800025e-06,
        2.1654152802774194e-06,
        2.1650220410060095e-06,
        2.109192097165465e-06,
        2.1280601845092366e-06,
        2.1854653440203226e-06,
        2.288951707536974e-06,
        2.3487056644142943e-06,
        2.318286830180978e-06,
        2.362444816408874e-06,
        2.383225237769734e-06,
        2.377162786727253e-06,
        2.348556679643509e-06,
        2.2858473644211508e-06,
        2.1986506517170503e-06,
        2.142185832753633e-06,
        2.1187004941070057e-06,
        2.06873121802442e-06,
        1.9832762834441474e-06,
        1.9330541931514867e-06,
        1.9059285230051612e-06,
        1.8706881746510427e-06,
        1.866765135424425e-06,
        1.8387798200410268e-06,
        1.7960858550936123e-06,
        1.7662808976901972e-06,
        1.7271101697085275e-06,
        1.6622382594419398e-06,
        1.6013575551129797e-06,
        1.5714300940284351e-06,
        1.5534352361642657e-06,
        1.5266935090848231e-06,
        1.5021980183405136e-06,
        1.4775389029399456e-06,
        1.475063755751624e-06,
        1.492798663586751e-06,
        1.525181902039094e-06,
        1.5839351503222734e-06,
        1.6847270711498073e-06,
        1.798681705641764e-06,
        1.914637290875121e-06,
        1.9668264613189216e-06,
        1.989830394050152e-06,
        2.0526205682486784e-06,
        1.995358149666781e-06,
        1.9626918188734223e-06,
        1.9364244764563103e-06,
        1.8450031461581454e-06,
        1.7724915970083857e-06,
        1.764362082516432e-06,
        1.8047114726693866e-06,
        1.8153445876094672e-06,
        1.8660963487948997e-06,
        1.91494998213495e-06,
        1.9582535931572128e-06,
        1.97237456874951e-06,
        1.90425474792677e-06,
        1.8690014208927406e-06,
        1.8123337875516936e-06,
        1.7737437450217912e-06,
        1.7441779603029706e-06,
        1.7134211648454078e-06,
        1.71058567516578e-06,
        1.7030553485185818e-06,
        1.7015442652026843e-06,
        1.7129343579661139e-06,
        1.7018594850956114e-06,
        1.6879397199583315e-06,
        1.6739808878488604e-06,
        1.6801284122560055e-06,
        1.6668760753239984e-06,
        1.6674199933048693e-06,
        1.660991561492104e-06,
        1.642375961027261e-06,
        1.6609182065824417e-06,
        1.6952521504453807e-06,
        1.7410456221909456e-06,
        1.772152702890375e-06,
        1.7924261701399477e-06,
        1.7999995817491466e-06,
        1.8619017801999743e-06,
        1.9471292196342614e-06,
        2.013317966731845e-06,
        2.0512635187864825e-06,
        2.1196672420948217e-06,
        2.1742180658355888e-06,
        2.205630759258831e-06,
        2.159132338975462e-06,
        2.0780090592737676e-06,
        2.0940258468435933e-06,
        2.0688715121532967e-06,
        1.9790174106457274e-06,
        1.8925015876402496e-06,
        1.8456916053895038e-06,
        1.8216803274251616e-06,
        1.8207512143357773e-06,
        1.8577941098408483e-06,
        1.941532053448271e-06,
        2.0304369601006918e-06,
        2.0918411733136137e-06,
        2.059887884320626e-06,
        1.996844249433013e-06,
        2.017887804118303e-06,
        2.033456794498076e-06,
        1.9709993442254704e-06,
        1.923515460796054e-06,
        1.8632131260510079e-06,
        1.814176563700405e-06,
        1.8798178576407472e-06,
        1.9217027336464816e-06,
        1.9728287652634535e-06,
        1.951619171854708e-06,
        1.9234652891147578e-06,
        1.939858572967911e-06,
        1.9090549446067345e-06,
        1.844615348767027e-06,
        1.8155031058998588e-06,
        1.8742102332246648e-06,
        1.8766276964819055e-06,
        1.8028940892342726e-06,
        1.7471354708502886e-06,
        1.6886075409609788e-06,
        1.6043901963414238e-06,
        1.5440301531154605e-06,
        1.4960467050753614e-06,
        1.4539977461497659e-06,
        1.4183324237697012e-06,
        1.3998648997317147e-06,
        1.3785646437789992e-06,
        1.3698477335467998e-06,
        1.3797496643577817e-06,
        1.3763791191880418e-06,
        1.3762045862213663e-06,
        1.3817121455063302e-06,
        1.3794218913154327e-06,
        1.379656407714485e-06,
        1.3953757167394598e-06,
        1.4158007301260287e-06,
        1.4397621146405483e-06,
        1.4563794239295462e-06,
        1.479462243466794e-06,
        1.4869752215632912e-06,
        1.4979328280509892e-06,
        1.504066064453642e-06,
        1.506772986642674e-06,
        1.5115345746140494e-06,
        1.5286273078417643e-06,
        1.5239120208608918e-06,
        1.5257389950985637e-06,
        1.5299584490438104e-06,
        1.5347947012494299e-06,
        1.5529589611645716e-06,
        1.5887323846553726e-06,
        1.6334765309484411e-06,
        1.663673172800325e-06,
        1.7165746076330262e-06,
        1.7277308550262506e-06,
        1.7151084668833786e-06,
        1.6943841746306597e-06,
        1.6784491298310944e-06,
        1.6449771798061963e-06,
        1.6357979797974856e-06,
        1.67873281955145e-06,
        1.6574970805674738e-06,
        1.642707768425976e-06,
        1.5971254907353598e-06,
        1.5581617260535867e-06,
        1.554810395222076e-06,
        1.5507912679377546e-06,
        1.5549801737038663e-06,
        1.5646484473600522e-06,
        1.54950560121507e-06,
        1.5304956649194813e-06,
        1.5163352377408961e-06,
        1.4757636306566743e-06,
        1.4428486591253443e-06,
        1.4194393922368862e-06,
        1.3941525866320855e-06,
        1.3855139576432853e-06
    ],
    "valid_loss": [
        0.00013683467650906238,
        0.00010612467323511977,
        5.875100102393449e-05,
        2.754197932405846e-05,
        2.3419927932928505e-05,
        2.181252448154572e-05,
        2.0932323699608878e-05,
        2.133761870803234e-05,
        1.9988624079727213e-05,
        2.0928201291164933e-05,
        2.046136357572098e-05,
        1.888836367516918e-05,
        1.883676384808594e-05,
        1.844289099451159e-05,
        1.8833635609796506e-05,
        1.8151111284900145e-05,
        1.785791946872786e-05,
        1.746274017108638e-05,
        1.7229477057523338e-05,
        1.9232780982526935e-05,
        1.7101204950552556e-05,
        1.630168671329682e-05,
        1.5879790320244583e-05,
        1.5788056225037083e-05,
        1.5399110621726747e-05,
        1.6251669270436204e-05,
        1.5852688479805633e-05,
        1.6046154738188996e-05,
        1.7711143266646885e-05,
        1.6462707712159085e-05,
        1.5757997602362614e-05,
        1.524867767248914e-05,
        1.5025194231554308e-05,
        1.4362871082824074e-05,
        1.4027176037562744e-05,
        1.3994645933808782e-05,
        1.3902480350685541e-05,
        1.3932379694024707e-05,
        1.3788629799314512e-05,
        1.4198185677256377e-05,
        1.389706033611539e-05,
        1.3846095662386494e-05,
        1.4272455215604935e-05,
        1.3243713982180168e-05,
        1.2892392082575855e-05,
        1.3162926953397487e-05,
        1.3248884177620228e-05,
        1.3200537301992777e-05,
        1.2953215577603894e-05,
        1.2881045737619994e-05,
        1.2734551760452501e-05,
        1.3067219712748299e-05,
        1.318489395016007e-05,
        1.2537714907799726e-05,
        1.2548184921813382e-05,
        1.2796689948292533e-05,
        1.298242968592348e-05,
        1.3026461093197488e-05,
        1.3446355987228658e-05,
        1.326686447743171e-05,
        1.2980759707886375e-05,
        1.2537193272886575e-05,
        1.305825529107058e-05,
        1.2867735602191004e-05,
        1.2676563476872616e-05,
        1.3031391878637872e-05,
        1.267330247307065e-05,
        1.3777981455292895e-05,
        1.3183182335601281e-05,
        1.2279945558847531e-05,
        1.2656098537269597e-05,
        1.2679247854129014e-05,
        1.2057128105860931e-05,
        1.2772041716652548e-05,
        1.2989656175618331e-05,
        1.3043621624888674e-05,
        1.2945339401051848e-05,
        1.3154387342080141e-05,
        1.2601958489566352e-05,
        1.2905597144707203e-05,
        1.312186214829336e-05,
        1.310751502779605e-05,
        1.245955412226204e-05,
        1.3142340443007106e-05,
        1.3441211127219246e-05,
        1.376660840011558e-05,
        1.2578743771944103e-05,
        1.2830280212159903e-05,
        1.3381351556947895e-05,
        1.3736937075652234e-05,
        1.315782569389663e-05,
        1.2612169453711612e-05,
        1.3005572325227534e-05,
        1.234814323538318e-05,
        1.2203485978738207e-05,
        1.2488841683690631e-05,
        1.2691868433767309e-05,
        1.3184534540562527e-05,
        1.2389775436294577e-05,
        1.2449390293801701e-05,
        1.2991322618479066e-05,
        1.2776899638179986e-05,
        1.3136816337132745e-05,
        1.2160020985278035e-05,
        1.2583423167064892e-05,
        1.1510171902765456e-05,
        1.120973651021254e-05,
        1.1220954606825668e-05,
        1.1634648997903825e-05,
        1.1287197920025176e-05,
        1.1196977469499783e-05,
        1.1361305233058121e-05,
        1.1624378721355035e-05,
        1.1486454600902713e-05,
        1.184065688365323e-05,
        1.2502378659603315e-05,
        1.2382462531177366e-05,
        1.2231363593991522e-05,
        1.255887293836981e-05,
        1.3503247580539321e-05,
        1.393060935625845e-05,
        1.2711312492994355e-05,
        1.2541835155858113e-05,
        1.2443755222669085e-05,
        1.2840872189758297e-05,
        1.2116346434418635e-05,
        1.2276053526063653e-05,
        1.3519391552625654e-05,
        1.3565688811553675e-05,
        1.2465979305313187e-05,
        1.1870614164605113e-05,
        1.214813002676262e-05,
        1.2220307919092693e-05,
        1.2090177880541176e-05,
        1.2405755415477114e-05,
        1.1858468476992426e-05,
        1.1890970888531495e-05,
        1.2031185606872417e-05,
        1.1865060795326365e-05,
        1.2273034681842982e-05,
        1.2795743895816381e-05,
        1.2561006024511289e-05,
        1.2757221078920446e-05,
        1.24987859384187e-05,
        1.2931287074983999e-05,
        1.3579137408057128e-05,
        1.411169798017956e-05,
        1.2011912218104532e-05,
        1.1818589330762812e-05,
        1.1684557046695682e-05,
        1.1557123954916543e-05,
        1.1502551240905433e-05,
        1.1867434077062269e-05,
        1.2080498371243426e-05,
        1.391368194980895e-05,
        1.3129854789266905e-05,
        1.1839896427936465e-05,
        1.1890415473044145e-05,
        1.163493299040549e-05,
        1.287547528165546e-05,
        1.31133150738259e-05,
        1.2416515510351735e-05,
        1.2170678363639259e-05,
        1.296949801076469e-05,
        1.2836953839550337e-05,
        1.2185657691510605e-05,
        1.220453239094351e-05,
        1.2214201294712153e-05,
        1.1807092348013887e-05,
        1.2550877940612018e-05,
        1.2686659547785876e-05,
        1.2870820436370568e-05,
        1.3091368306128096e-05,
        1.3805789152613574e-05,
        1.2044575676567089e-05,
        1.2057335895671969e-05,
        1.1736645710111318e-05,
        1.1434875592080312e-05,
        1.1561498539274828e-05,
        1.2033555746229324e-05,
        1.2899276052960259e-05,
        1.2675356017743823e-05,
        1.2898437430565995e-05,
        1.1704812428899476e-05,
        1.1975234174077292e-05,
        1.2492995515924526e-05,
        1.2725664719857529e-05,
        1.1700529759126794e-05,
        1.1149360804191274e-05,
        1.157396789192399e-05,
        1.2010955167301895e-05,
        1.2127787639940402e-05,
        1.1455766520435827e-05,
        1.10649117254874e-05,
        1.1453646593006388e-05,
        1.173806842220126e-05,
        1.176877633893555e-05,
        1.2152814528249275e-05,
        1.24761971434125e-05,
        1.1365107904439331e-05,
        1.1260096668782288e-05,
        1.1645714492737015e-05,
        1.1383026535063039e-05,
        1.1258277624141612e-05,
        1.1731924482070158e-05,
        1.1918968110585303e-05,
        1.1798147959002258e-05,
        1.1718804853367733e-05,
        1.1600832482748825e-05,
        1.1675430792499725e-05,
        1.205020976570627e-05,
        1.1614317413009404e-05,
        1.1767790613924586e-05,
        1.2083054696555784e-05,
        1.2137811436136781e-05,
        1.2162931024626006e-05,
        1.213114193311878e-05,
        1.2187865998349276e-05,
        1.2033234437977098e-05,
        1.2029294483913545e-05,
        1.192308187748701e-05,
        1.1614593353164895e-05,
        1.1585526740259383e-05,
        1.1606225001502771e-05,
        1.1471697596345255e-05,
        1.1536611683229917e-05,
        1.2017849154019344e-05,
        1.2373957289630278e-05,
        1.3594981086550762e-05,
        1.2420498082930406e-05,
        1.2660741991430634e-05,
        1.238570075273162e-05,
        1.2047390266153089e-05,
        1.1883255366105562e-05,
        1.2362049833625141e-05,
        1.2678934009026899e-05,
        1.2900171630973809e-05,
        1.2131468740534251e-05,
        1.2366979637072089e-05,
        1.2876051908201028e-05,
        1.2417648141580713e-05,
        1.2292343225975873e-05,
        1.2197902167662948e-05,
        1.1902537200016996e-05,
        1.2385217022765089e-05,
        1.2904783661344894e-05,
        1.335177489665177e-05,
        1.256423599732068e-05,
        1.3255905234288285e-05,
        1.4044726617046672e-05,
        1.356340980118762e-05,
        1.3829035883221835e-05,
        1.4836914076904664e-05,
        1.2614953405102413e-05,
        1.3341049171547405e-05,
        1.264726079274511e-05,
        1.2246061089746766e-05,
        1.1824395661550653e-05,
        1.1975324124676022e-05,
        1.1692173387784588e-05,
        1.1470154491860069e-05,
        1.1203624975864818e-05,
        1.1372680841418371e-05,
        1.1997472790224247e-05,
        1.2289228735594546e-05,
        1.1804583551183828e-05,
        1.14811948476613e-05,
        1.1568497402891234e-05,
        1.21321620279e-05,
        1.211397609866304e-05,
        1.2248473062024041e-05,
        1.2327250932236858e-05,
        1.1977332890448516e-05,
        1.1891306337489201e-05,
        1.1852492064941488e-05,
        1.219106572576084e-05,
        1.190514890975914e-05,
        1.2206567474140091e-05,
        1.2119906749819864e-05,
        1.1791315248675207e-05,
        1.1534060071486051e-05,
        1.1387829661356755e-05,
        1.1442744305482246e-05,
        1.2235973267578364e-05,
        1.236658919648197e-05,
        1.1903134448424717e-05,
        1.1857312867117049e-05,
        1.1950225747245015e-05,
        1.198539171777964e-05,
        1.2429757299037584e-05,
        1.2624026238855457e-05,
        1.3003995440168151e-05,
        1.2220167101233983e-05,
        1.1616952494195321e-05,
        1.1446739643975582e-05,
        1.1314722017641562e-05,
        1.1845684297247709e-05,
        1.1527370339403268e-05,
        1.1401341498237478e-05,
        1.1894356016303758e-05,
        1.2125605454127128e-05,
        1.1731632044424944e-05,
        1.1471531639454588e-05,
        1.1527259766942387e-05,
        1.1541153010073617e-05,
        1.150687417240899e-05,
        1.157929481351642e-05,
        1.2101776794208751e-05,
        1.203442854199516e-05,
        1.1712324678684177e-05,
        1.1608532489678468e-05,
        1.2193196455118075e-05,
        1.2177847897714827e-05,
        1.1648348788328183e-05,
        1.1399100982014111e-05,
        1.1296325549011942e-05,
        1.152204341781084e-05,
        1.2163286506249803e-05,
        1.2381765315837873e-05,
        1.183477061859971e-05,
        1.1472524042020916e-05,
        1.1540581882691293e-05,
        1.1901088563300346e-05,
        1.1968366897581298e-05,
        1.2198490774528429e-05,
        1.2247089040475473e-05,
        1.2203822802486726e-05,
        1.2495456784272284e-05,
        1.1674901498037772e-05,
        1.1956139704513429e-05,
        1.2364835356205438e-05,
        1.2210071030320723e-05,
        1.1505621345183457e-05,
        1.1886029301163315e-05,
        1.1845013595730986e-05,
        1.1847721933627218e-05,
        1.151178433598722e-05,
        1.1616862739995279e-05,
        1.1972057818109504e-05,
        1.2178000892292141e-05,
        1.1781360388418031e-05,
        1.160649838847533e-05,
        1.1478380061676613e-05,
        1.1825248621049081e-05,
        1.2665880566682054e-05,
        1.2092812961727092e-05,
        1.1695433409593116e-05,
        1.2094088571200338e-05,
        1.2348232203988474e-05,
        1.205844947622829e-05,
        1.2304680402308564e-05,
        1.177897512636221e-05,
        1.1879342907858215e-05,
        1.1912395039322708e-05,
        1.1671896598123891e-05,
        1.2337089917268596e-05,
        1.2877754684818892e-05,
        1.218257717810216e-05,
        1.2002425572317595e-05,
        1.1541182469876694e-05,
        1.1557656391757492e-05,
        1.1912665873112332e-05,
        1.1986728800041975e-05,
        1.2192450925701532e-05,
        1.268829908402647e-05,
        1.2668422554890246e-05,
        1.1980369410551032e-05,
        1.184043475673803e-05,
        1.2025757147158715e-05,
        1.1885086391066158e-05,
        1.1540052391830652e-05,
        1.1750484158008828e-05,
        1.1982427275995321e-05,
        1.2421935132124513e-05,
        1.256983669868304e-05,
        1.2153022514459002e-05,
        1.1715845321550595e-05,
        1.1775906396874995e-05,
        1.1738219845589079e-05,
        1.1798631099772726e-05,
        1.1750409722906384e-05,
        1.1737427966082363e-05,
        1.1697789604643234e-05,
        1.1668220211098538e-05,
        1.1809683435893875e-05,
        1.1566252173099374e-05,
        1.139755257476437e-05,
        1.1917013158053097e-05,
        1.2813786454011699e-05,
        1.2413495094941569e-05,
        1.1724598025443534e-05,
        1.1918829845909527e-05,
        1.2113827621255529e-05,
        1.2539961119585026e-05,
        1.2080829303031328e-05,
        1.1649240438367988e-05,
        1.1694185295936076e-05,
        1.1744799987204415e-05,
        1.2053523207957712e-05,
        1.1629266495482925e-05,
        1.1464545541752847e-05,
        1.1893897621767877e-05,
        1.2011160207531312e-05,
        1.1808895680759588e-05,
        1.1519624964376885e-05,
        1.1621014411843612e-05,
        1.1963424917415746e-05,
        1.1824582633100849e-05,
        1.1508345002177292e-05,
        1.1503201320560005e-05,
        1.1763917828212052e-05,
        1.1960860539757214e-05,
        1.221544332000989e-05,
        1.2062250969217378e-05,
        1.2180895808941198e-05,
        1.2198593883839202e-05,
        1.2182979206214822e-05,
        1.2019588460793028e-05,
        1.1684561956662861e-05,
        1.1681714568496104e-05,
        1.1572657716281802e-05,
        1.1928017769293261e-05,
        1.2375729002187344e-05,
        1.2444997444365509e-05,
        1.24331339816663e-05,
        1.232645139318134e-05,
        1.215687271432251e-05,
        1.1993644587013702e-05,
        1.1747321942746513e-05,
        1.1499246047398852e-05,
        1.1600451665694379e-05,
        1.2277280232263787e-05,
        1.3188175183026815e-05,
        1.2400707772817859e-05,
        1.1441286634425984e-05,
        1.1729922983049089e-05,
        1.242418546828224e-05,
        1.240927762953303e-05,
        1.1873530488711076e-05,
        1.229332050584329e-05,
        1.1794213307703262e-05,
        1.1813674453616093e-05,
        1.2144887877234624e-05,
        1.2119342103594217e-05,
        1.1803079726436077e-05,
        1.1610771238313653e-05,
        1.2004644877482749e-05,
        1.2303233533180098e-05,
        1.19206443733804e-05,
        1.2211415182935793e-05,
        1.2134999006936341e-05,
        1.1895021218657242e-05,
        1.1702463107803409e-05,
        1.156112282858625e-05,
        1.1996469782128812e-05,
        1.2107572323068794e-05,
        1.1982841677225275e-05,
        1.2163034133936775e-05,
        1.181294149371553e-05,
        1.1685082020386521e-05,
        1.2001558079316313e-05,
        1.2487189381535372e-05,
        1.1915495388998557e-05,
        1.1620839420613332e-05,
        1.1858653680954439e-05,
        1.1752532203518755e-05,
        1.169355780213053e-05,
        1.1832798775780406e-05,
        1.1971673465878689e-05,
        1.2840255694279232e-05,
        1.2846566966091816e-05,
        1.2119815424430325e-05,
        1.1826758730554821e-05,
        1.2181384645273593e-05,
        1.257328742361682e-05,
        1.2337838981861507e-05,
        1.2218089399522289e-05,
        1.2178814572053134e-05,
        1.210178445375755e-05,
        1.1931212586737645e-05,
        1.1657700311818339e-05,
        1.1781247066375526e-05,
        1.2160951522257901e-05,
        1.2666038274827862e-05,
        1.240271457460348e-05,
        1.1657475435321516e-05,
        1.150911115345599e-05,
        1.1269412447711377e-05,
        1.1549530788872725e-05,
        1.2404436794691374e-05,
        1.2411517752959023e-05,
        1.1749618628994416e-05,
        1.1302193745386243e-05,
        1.1255406864531076e-05,
        1.156272760225921e-05,
        1.1777287472443256e-05,
        1.1615971089955474e-05,
        1.1787452675694405e-05,
        1.1871574554185431e-05,
        1.2012690546101833e-05,
        1.201744614391325e-05,
        1.2150459118793907e-05,
        1.1760982257034747e-05,
        1.1559815598924368e-05,
        1.1856100694419763e-05,
        1.2251874294488656e-05,
        1.2124438649526581e-05,
        1.1790652403105969e-05,
        1.1789043308661889e-05,
        1.20241026846179e-05,
        1.1774601327598673e-05,
        1.1962878536268007e-05,
        1.2802260011459693e-05,
        1.2551692995163823e-05,
        1.2337183599442382e-05,
        1.2169192607570731e-05,
        1.183218385149084e-05,
        1.1733560679533069e-05,
        1.2244213370897758e-05,
        1.2210645103683356e-05,
        1.1951038052215199e-05,
        1.2042768808645018e-05,
        1.3172089541350573e-05,
        1.3034537792809147e-05,
        1.1965831390529781e-05,
        1.1955807790732091e-05,
        1.2061304131146476e-05,
        1.1688363056854576e-05,
        1.1801917635404023e-05,
        1.2138979419129452e-05,
        1.2213071805862171e-05,
        1.2302876676765487e-05,
        1.2494610895126593e-05,
        1.1836208649787255e-05,
        1.1827599513334647e-05,
        1.220828871223455e-05,
        1.2219371490152211e-05,
        1.248073729186408e-05,
        1.1858483010495277e-05,
        1.1631221251616445e-05,
        1.195306586866035e-05,
        1.2538948291555231e-05,
        1.2552990994087405e-05,
        1.169386104170354e-05,
        1.1666931050115876e-05,
        1.2358653511127708e-05,
        1.2671150729053888e-05,
        1.2058944793717362e-05,
        1.149248266580838e-05,
        1.1345103126960382e-05,
        1.1839139703794754e-05,
        1.2376800749823294e-05,
        1.2066796813230885e-05,
        1.1790704448758074e-05,
        1.1762768503094666e-05,
        1.1873816445199611e-05,
        1.1804769737139277e-05,
        1.1513759124786831e-05,
        1.1478567033226809e-05,
        1.2117318018724123e-05,
        1.2556699197700084e-05,
        1.183264813798734e-05,
        1.1527617408951745e-05,
        1.179808864659873e-05,
        1.2538384234525644e-05,
        1.2571852927605645e-05,
        1.2012774408341258e-05,
        1.2723487444011436e-05,
        1.2794190774998148e-05,
        1.2263416645334322e-05,
        1.1843417856397632e-05,
        1.1858832600158459e-05,
        1.2113685821403385e-05,
        1.2046177897057118e-05,
        1.1933348226062058e-05,
        1.1744035996311276e-05,
        1.1621964785090884e-05,
        1.1641169827115631e-05,
        1.1722713187242651e-05,
        1.1756964136293697e-05,
        1.1678421551708128e-05,
        1.169043820538334e-05,
        1.2030498800663344e-05,
        1.1949586273119553e-05,
        1.1736760406944631e-05,
        1.1551364956212317e-05,
        1.1551393041224582e-05,
        1.2032664292588208e-05,
        1.2681399205348402e-05,
        1.2592514265895845e-05,
        1.2135827023801498e-05,
        1.1802508991851125e-05,
        1.1900648826639747e-05,
        1.1857427171152988e-05,
        1.1907958785776646e-05,
        1.2006612399530934e-05,
        1.234016905588623e-05,
        1.2026360680324425e-05,
        1.1925424717426401e-05,
        1.1929523950825255e-05,
        1.2085387912959503e-05,
        1.1835108227942976e-05,
        1.1989212654238762e-05,
        1.1944691232240235e-05,
        1.2098208426461345e-05,
        1.1885855291926474e-05,
        1.1819592535256937e-05,
        1.2068224435288009e-05,
        1.2278063863025643e-05,
        1.2407519664884065e-05,
        1.2325557386357289e-05,
        1.2035515215931336e-05,
        1.1789452210728604e-05,
        1.17870636098951e-05,
        1.176657647724043e-05,
        1.2287613945588539e-05,
        1.301428889176203e-05,
        1.2834519477822718e-05,
        1.190566720589461e-05,
        1.1487679343115977e-05,
        1.167887503627683e-05,
        1.2024080491566246e-05,
        1.2163723886126159e-05,
        1.1998191413020646e-05,
        1.1926884745266908e-05,
        1.198076318991883e-05,
        1.1879954296971414e-05,
        1.1759627695289254e-05,
        1.2194213603918989e-05,
        1.2495797339595858e-05,
        1.208183957787819e-05,
        1.1785592976525483e-05,
        1.1929807746928233e-05,
        1.2534159895163054e-05,
        1.232046457199998e-05,
        1.24594091800309e-05,
        1.251191145908174e-05,
        1.2366764384110937e-05,
        1.1951034909836203e-05,
        1.1964656337184376e-05,
        1.2375178496667174e-05,
        1.249503963346071e-05,
        1.1739721706349958e-05,
        1.1325855466420517e-05,
        1.187941105820267e-05,
        1.2696102985861633e-05,
        1.2233741982093293e-05,
        1.1582654998655412e-05,
        1.1494371628381692e-05,
        1.170588731891442e-05,
        1.208387858904851e-05,
        1.2158550155509729e-05,
        1.1657317530777022e-05,
        1.1742086346543624e-05,
        1.1880685096486416e-05,
        1.1896416042133608e-05,
        1.221098212383056e-05,
        1.2243460574729791e-05,
        1.1888987458189647e-05,
        1.1519881461062343e-05,
        1.1731330965237494e-05,
        1.227188299994135e-05,
        1.199171929068326e-05,
        1.1412450593579223e-05,
        1.135285517954213e-05,
        1.1754208269915165e-05,
        1.2042918857242026e-05,
        1.2261600153876579e-05,
        1.2148403610133863e-05,
        1.2194558283614993e-05,
        1.2256530318165669e-05,
        1.225561038671491e-05,
        1.208539085893981e-05,
        1.1803202475615564e-05,
        1.1772467455862445e-05,
        1.19420840396679e-05,
        1.2317636430904565e-05,
        1.2221476884078798e-05,
        1.1950343979254698e-05,
        1.1884684559352184e-05,
        1.2148891071675449e-05,
        1.2645834349080109e-05,
        1.2820955988087267e-05,
        1.2353275133079236e-05,
        1.1990352552219164e-05,
        1.136440008357073e-05,
        1.1679146066465141e-05,
        1.2331107416858356e-05,
        1.200251650490976e-05,
        1.1970406890745054e-05,
        1.1923310092761514e-05,
        1.1679676735717905e-05,
        1.198296521199951e-05,
        1.1908492008212346e-05,
        1.2757334597361638e-05,
        1.2593441267699343e-05,
        1.1468142583708579e-05,
        1.13716905992376e-05,
        1.1782908206471708e-05,
        1.230160538806336e-05,
        1.2062092868274198e-05,
        1.1509218779936563e-05,
        1.1615359111646217e-05,
        1.2207334214614846e-05,
        1.2358469878355193e-05,
        1.2114108078580827e-05,
        1.1760457676141286e-05,
        1.1613937185151023e-05,
        1.1581992938680923e-05,
        1.1743541464416954e-05,
        1.2006316426709352e-05,
        1.209844371208859e-05,
        1.1930477662850211e-05,
        1.168758021168747e-05,
        1.1714959367072717e-05,
        1.2118864265588304e-05,
        1.2570135421086244e-05,
        1.2281615733283323e-05,
        1.1944882720960237e-05,
        1.1992672020714778e-05,
        1.1872258021616826e-05,
        1.2057878152447277e-05,
        1.1961045350921853e-05,
        1.2241682381016049e-05,
        1.2454038657929916e-05,
        1.2106092655359566e-05,
        1.1843055500819782e-05,
        1.2198017846489698e-05,
        1.2329078422021081e-05,
        1.2158486325936395e-05,
        1.2218469430981987e-05,
        1.1654676164833114e-05,
        1.1548052692352997e-05,
        1.1706697463499047e-05,
        1.208223119686043e-05,
        1.2268386317714765e-05,
        1.2207047079734188e-05,
        1.1856214998455703e-05,
        1.1687137529046563e-05,
        1.180032700243654e-05,
        1.1766697458831734e-05,
        1.183197213370606e-05,
        1.201922374843093e-05,
        1.2503442551291778e-05,
        1.2369949774418336e-05,
        1.2372417327524085e-05,
        1.2505758288212337e-05,
        1.1876580756721694e-05,
        1.1818585009991695e-05,
        1.192787066667656e-05,
        1.2163977436831308e-05,
        1.2468597496212005e-05,
        1.2413626485663293e-05,
        1.2027750397434923e-05,
        1.1851556814393129e-05,
        1.1901252359805455e-05,
        1.2164724144639974e-05,
        1.2275278340445346e-05,
        1.2175101851270652e-05,
        1.2175105190048333e-05,
        1.1943921152987796e-05,
        1.1666264472971582e-05,
        1.1869669094122397e-05,
        1.236776130384707e-05,
        1.2244344565220797e-05,
        1.2071069859467896e-05,
        1.1925020332529494e-05,
        1.1778272215460786e-05,
        1.196301562255166e-05,
        1.2001840893425855e-05,
        1.1956571388827856e-05,
        1.1882024928330371e-05,
        1.1974395748081713e-05,
        1.204620342888645e-05,
        1.19332602394502e-05,
        1.1858826904596532e-05,
        1.1784474486001983e-05,
        1.1793809315603727e-05,
        1.1875302397666829e-05,
        1.194551178595528e-05,
        1.2217666553148786e-05,
        1.2531171885536272e-05,
        1.250920056800257e-05,
        1.2166321455162822e-05,
        1.1764926139072042e-05,
        1.1639097035371117e-05,
        1.1993923473149501e-05,
        1.2577200667458917e-05,
        1.2651081336406856e-05,
        1.2310895831561801e-05,
        1.1893510323556753e-05,
        1.1767050780069973e-05,
        1.1898888701605223e-05,
        1.2472447696075516e-05,
        1.2046562642085306e-05,
        1.224415209450736e-05,
        1.2410641029219443e-05,
        1.2258914205430681e-05,
        1.2296506681745438e-05,
        1.240258907584237e-05,
        1.1921648756266646e-05,
        1.162120491857018e-05,
        1.1977357243885726e-05,
        1.2251986634537721e-05,
        1.2052383506375996e-05,
        1.1700328450472434e-05,
        1.1636615537958576e-05,
        1.185570239788216e-05,
        1.1845867340824161e-05,
        1.1585397509923217e-05,
        1.1621445114164603e-05,
        1.1967693642881639e-05,
        1.2301269742706968e-05,
        1.2289218326464126e-05,
        1.1998236584718697e-05,
        1.19017573008302e-05,
        1.1910660642516205e-05,
        1.1939747877283874e-05,
        1.1753617502664122e-05,
        1.1856048255970285e-05,
        1.2263566104735264e-05,
        1.2158091564575159e-05,
        1.1982994868201277e-05,
        1.2193491249547534e-05,
        1.2893076139203308e-05,
        1.2505160254209869e-05,
        1.24409158868485e-05,
        1.220804419586901e-05,
        1.2268623763727568e-05,
        1.266785299869742e-05,
        1.2837510826227185e-05,
        1.2465022254510553e-05,
        1.242968384592858e-05,
        1.2187272481516614e-05,
        1.2359025097443856e-05,
        1.2155612423946867e-05,
        1.2217286128891717e-05,
        1.1881277631325644e-05,
        1.1732097902110939e-05,
        1.193837446126441e-05,
        1.1979782178476358e-05,
        1.2320990331285565e-05,
        1.2795755679737612e-05,
        1.2110909725960073e-05,
        1.1651003705781505e-05,
        1.1821454002014046e-05,
        1.2164398515616627e-05,
        1.2106350134038462e-05,
        1.1604136497863281e-05,
        1.1647987218345082e-05,
        1.1922631928094676e-05,
        1.1986653579344784e-05,
        1.1689146294819055e-05,
        1.177106497283728e-05,
        1.1910058484141307e-05,
        1.1719956535269368e-05,
        1.1557527554218702e-05,
        1.169005974511314e-05,
        1.1952429144116507e-05,
        1.1980673435718788e-05,
        1.2009033013350447e-05,
        1.232017959750488e-05,
        1.2642233182751947e-05,
        1.218936471673116e-05,
        1.1915768972369801e-05,
        1.1947177443221267e-05,
        1.2037093475781528e-05,
        1.2546694255777675e-05,
        1.2009554451864915e-05,
        1.1566429128316524e-05,
        1.1739491134291206e-05,
        1.2649623272553218e-05,
        1.2505486079631904e-05,
        1.193342541074612e-05,
        1.1656837532385548e-05,
        1.2020701648551973e-05,
        1.2922473291099347e-05,
        1.2750820052907819e-05,
        1.18968512616244e-05,
        1.170324084660465e-05,
        1.1831361726586299e-05,
        1.2083394662683294e-05,
        1.2264327149648095e-05,
        1.204192056271508e-05,
        1.1818286680385868e-05,
        1.1699124526520009e-05,
        1.1790974300554261e-05,
        1.2203220644111826e-05,
        1.2188479744246718e-05,
        1.1985556889075562e-05,
        1.1964124685938175e-05,
        1.1861102772583594e-05,
        1.1798431165909174e-05,
        1.2032610282949235e-05,
        1.2164242378660317e-05,
        1.2259967295191349e-05,
        1.2398968466044175e-05,
        1.1819317773493569e-05,
        1.2050627505713904e-05,
        1.2491706747739236e-05,
        1.2362435757045452e-05,
        1.2305302789748242e-05,
        1.2406600911825429e-05,
        1.2353087965130353e-05,
        1.2354763049533326e-05,
        1.2689519505468615e-05,
        1.2354906616973655e-05,
        1.2785070412762806e-05,
        1.291451639468687e-05,
        1.2480015723087375e-05,
        1.2250019112489535e-05,
        1.1913256640363372e-05,
        1.2224765383696968e-05,
        1.2244458280060674e-05,
        1.2205932320785745e-05,
        1.2179202263061632e-05,
        1.1965819017412489e-05,
        1.2215863416801772e-05,
        1.2410169279572833e-05,
        1.2280473871316049e-05,
        1.2266907042802915e-05,
        1.230297055533796e-05,
        1.2705362005570124e-05,
        1.2741589511008968e-05,
        1.2210431814709075e-05,
        1.1889498291175007e-05,
        1.1871671378738213e-05,
        1.2043651227946524e-05,
        1.2300199762659201e-05,
        1.2308997441851505e-05,
        1.2229131915709079e-05,
        1.2447389973172757e-05,
        1.2617136769307812e-05,
        1.2492332866753975e-05,
        1.2265736899424685e-05,
        1.1609995070701912e-05,
        1.1907961535358267e-05,
        1.258789104079959e-05,
        1.2292593437903342e-05,
        1.2102682977751406e-05,
        1.2075389844991143e-05,
        1.2382420894655684e-05,
        1.2053688772051006e-05,
        1.1590050194822551e-05,
        1.1458277674050133e-05,
        1.1647428267681364e-05,
        1.2213596975951694e-05,
        1.2341921128574579e-05,
        1.1709438403577354e-05,
        1.1477334435066057e-05,
        1.1619666134856109e-05,
        1.200387676221718e-05,
        1.2144686372181577e-05,
        1.19394073219603e-05,
        1.1728357881910938e-05,
        1.1545781341535741e-05,
        1.1576389094939569e-05,
        1.1843395466947292e-05,
        1.206440958718819e-05,
        1.2097224469038566e-05,
        1.2032478695828822e-05,
        1.210333227181123e-05,
        1.2205297167431398e-05,
        1.1963059026661526e-05,
        1.1718382792588982e-05,
        1.1786290191864979e-05,
        1.1923066361990722e-05,
        1.2027784963603864e-05,
        1.2232763523833755e-05,
        1.2193584735322631e-05,
        1.1973562428452002e-05,
        1.2070957519418827e-05,
        1.2460708357346605e-05,
        1.2359512951782814e-05,
        1.2145591770129482e-05,
        1.2141966643161482e-05,
        1.2066402837464398e-05,
        1.2454556757666701e-05,
        1.2594125520725483e-05,
        1.2457844078892749e-05,
        1.1957620943412151e-05,
        1.1725269119757633e-05,
        1.1884349503191852e-05,
        1.2225901157304939e-05,
        1.2105016979749873e-05,
        1.1804353175523763e-05,
        1.193676752720589e-05,
        1.201348871036654e-05,
        1.2145625157906304e-05,
        1.2413188712989565e-05,
        1.2295328486021034e-05,
        1.181729977698278e-05,
        1.1779799215253622e-05,
        1.1990641454688007e-05,
        1.1935897677420363e-05,
        1.1948608011258699e-05,
        1.1857934665360667e-05,
        1.1685256815218113e-05,
        1.1778679546338001e-05,
        1.1988322575388453e-05
    ],
    "test_loss": [
        0.00013724512148386112,
        0.00010715913340191658,
        5.8910575742864344e-05,
        2.7117723200779734e-05,
        2.2922231662957187e-05,
        2.1466353726708803e-05,
        2.056578482972173e-05,
        2.0795768478006212e-05,
        1.9471567612768277e-05,
        2.0390709540804745e-05,
        2.007010635986252e-05,
        1.854206918915934e-05,
        1.845691975273698e-05,
        1.8063172201525052e-05,
        1.838401891683939e-05,
        1.7741111938722626e-05,
        1.7463541870527458e-05,
        1.7095922218664747e-05,
        1.6899086544404095e-05,
        1.867491404676377e-05,
        1.672646015071758e-05,
        1.5872825040803688e-05,
        1.5524533572942583e-05,
        1.5433982387027977e-05,
        1.5055321004573228e-05,
        1.591896812676227e-05,
        1.5435026049651663e-05,
        1.5654871794102514e-05,
        1.731120267563206e-05,
        1.6034471373085936e-05,
        1.5390743644855435e-05,
        1.4954803174480211e-05,
        1.4716839258339475e-05,
        1.402911763498422e-05,
        1.3707233541804193e-05,
        1.3672492970826002e-05,
        1.3520977275625958e-05,
        1.3624139024845813e-05,
        1.3495952351303957e-05,
        1.383433982616786e-05,
        1.3598185527530815e-05,
        1.3557268021442074e-05,
        1.3941044607704472e-05,
        1.3017452481815153e-05,
        1.2627707732246676e-05,
        1.2888261621785734e-05,
        1.2971947887988595e-05,
        1.2919501189766229e-05,
        1.2641567980398462e-05,
        1.2630759571446792e-05,
        1.2581416758076644e-05,
        1.2756687856484746e-05,
        1.29282931733966e-05,
        1.2363678371539252e-05,
        1.2365062785885194e-05,
        1.2544456096338554e-05,
        1.2756727725418245e-05,
        1.276806248285156e-05,
        1.3190988201423317e-05,
        1.3023541037516472e-05,
        1.2757604449157823e-05,
        1.2365643340404503e-05,
        1.2856883789129469e-05,
        1.2588024984704246e-05,
        1.2378337765949178e-05,
        1.2819039726096435e-05,
        1.2442557387075963e-05,
        1.3482654981789636e-05,
        1.2848977563578287e-05,
        1.2096325748646021e-05,
        1.243001202813486e-05,
        1.2405985201941117e-05,
        1.183314424107116e-05,
        1.2476258223404216e-05,
        1.2700216752763338e-05,
        1.2796380816758911e-05,
        1.2664883646945921e-05,
        1.2750952032825606e-05,
        1.230814546434651e-05,
        1.2659149001678904e-05,
        1.291239725285218e-05,
        1.2882134375543043e-05,
        1.2255097786141367e-05,
        1.2961899149158945e-05,
        1.3236758129876257e-05,
        1.3508900523952469e-05,
        1.2405620882376395e-05,
        1.2661286997787563e-05,
        1.3166904026812916e-05,
        1.3401722225181064e-05,
        1.2798017210620507e-05,
        1.231077190399019e-05,
        1.2825586676333638e-05,
        1.2196257328657799e-05,
        1.2003848088008854e-05,
        1.2318295741297435e-05,
        1.2542899636742638e-05,
        1.2943306085443456e-05,
        1.2171075285386055e-05,
        1.2183908172005192e-05,
        1.2702490460364841e-05,
        1.2523789258885112e-05,
        1.2854849295128954e-05,
        1.190912166240345e-05,
        1.2246647929024064e-05,
        1.1224839176459436e-05,
        1.098625071249346e-05,
        1.0982777205311965e-05,
        1.1354732750991585e-05,
        1.1011208664863132e-05,
        1.091526692418145e-05,
        1.1020836521306153e-05,
        1.127198644890558e-05,
        1.1171538340346516e-05,
        1.1518876096182661e-05,
        1.2185897494307653e-05,
        1.214357534480819e-05,
        1.1966097510750915e-05,
        1.2213639005270752e-05,
        1.3169015901896182e-05,
        1.359274430190245e-05,
        1.2349474818482272e-05,
        1.2204569706694076e-05,
        1.2241187849121725e-05,
        1.2610076825699691e-05,
        1.1830960287669698e-05,
        1.2006056787644897e-05,
        1.3221283091718462e-05,
        1.3295072238878174e-05,
        1.2235522139793908e-05,
        1.1643260098343305e-05,
        1.1901386696507487e-05,
        1.1983968809291009e-05,
        1.183329546806029e-05,
        1.212331544543459e-05,
        1.1523944753501447e-05,
        1.1575775938238187e-05,
        1.177921414356451e-05,
        1.1627981458872695e-05,
        1.1950191573873444e-05,
        1.2488868001114713e-05,
        1.2240326444479746e-05,
        1.2424610082243926e-05,
        1.2210894726414762e-05,
        1.2564737403169056e-05,
        1.3313330835629396e-05,
        1.3845980572755806e-05,
        1.1719501086713793e-05,
        1.1515256271979216e-05,
        1.1405952743013818e-05,
        1.1295465322762086e-05,
        1.1263074662076025e-05,
        1.1587216161766519e-05,
        1.1820237115748267e-05,
        1.3605929134969015e-05,
        1.2745463475113629e-05,
        1.1623098398313298e-05,
        1.163092527879486e-05,
        1.1356561419167934e-05,
        1.2539186130365405e-05,
        1.2859076580471853e-05,
        1.2255159848126516e-05,
        1.1898765363229672e-05,
        1.2675930680302518e-05,
        1.2516632687122899e-05,
        1.191622324253325e-05,
        1.1964138237447591e-05,
        1.1910505094755957e-05,
        1.1511310033157674e-05,
        1.221756737181176e-05,
        1.2393177257955248e-05,
        1.2614083162519509e-05,
        1.2771187382363309e-05,
        1.3489651881419168e-05,
        1.1800140030886342e-05,
        1.18274653730313e-05,
        1.1478877932348616e-05,
        1.1178730067473734e-05,
        1.131740443091109e-05,
        1.1793569905604053e-05,
        1.2629136336297236e-05,
        1.2308659439710865e-05,
        1.2406499373704156e-05,
        1.1384829671410055e-05,
        1.1731122389831708e-05,
        1.2254791796986736e-05,
        1.2471536013369618e-05,
        1.1452024143451581e-05,
        1.0893858288886525e-05,
        1.1306550457463993e-05,
        1.170229086615475e-05,
        1.180658681779308e-05,
        1.1191450614043804e-05,
        1.0846614584684977e-05,
        1.1205950925517113e-05,
        1.1447734403326158e-05,
        1.1513761481571077e-05,
        1.1896868741107561e-05,
        1.2259580978973662e-05,
        1.1121489473695949e-05,
        1.101045547589779e-05,
        1.1369609951545598e-05,
        1.1155638295429662e-05,
        1.104753476244294e-05,
        1.148624975707198e-05,
        1.1684096295375555e-05,
        1.1541301683879812e-05,
        1.1461421231437163e-05,
        1.131714793422563e-05,
        1.1380691747469825e-05,
        1.167068128304761e-05,
        1.1409632861614227e-05,
        1.161786123092091e-05,
        1.1880987157667302e-05,
        1.1888664971545295e-05,
        1.1888132141906972e-05,
        1.1846242265917991e-05,
        1.1891148629343393e-05,
        1.175988379917734e-05,
        1.180832416057989e-05,
        1.1703604184175936e-05,
        1.1391048832238349e-05,
        1.1343766241096736e-05,
        1.1333698647993116e-05,
        1.1265552035116134e-05,
        1.1345689573440306e-05,
        1.1806343087022286e-05,
        1.2080937322309277e-05,
        1.3134115462386624e-05,
        1.2107545809246025e-05,
        1.2431513692497051e-05,
        1.2102631324896678e-05,
        1.1733206179902706e-05,
        1.154051098276522e-05,
        1.2114972232804424e-05,
        1.2478597331768548e-05,
        1.2591636167365455e-05,
        1.1832640871235913e-05,
        1.212124776005594e-05,
        1.2548478145053345e-05,
        1.2163218355905352e-05,
        1.2036044903190666e-05,
        1.189286535026805e-05,
        1.1610616476148154e-05,
        1.2072974926733557e-05,
        1.2675936179465758e-05,
        1.3148007134328356e-05,
        1.23021670883087e-05,
        1.3006090424964319e-05,
        1.3733552537076033e-05,
        1.3190737007502412e-05,
        1.3461579831465545e-05,
        1.4475478823302618e-05,
        1.2386967327865253e-05,
        1.31350243955109e-05,
        1.2520880397929264e-05,
        1.2083159573454739e-05,
        1.1641382134096475e-05,
        1.1661801312805378e-05,
        1.1413487971444917e-05,
        1.1193985531899256e-05,
        1.0947331563046792e-05,
        1.1118676651698134e-05,
        1.1734505357218412e-05,
        1.206199997169516e-05,
        1.1572916766150197e-05,
        1.1232100232323228e-05,
        1.1267095925196069e-05,
        1.1831096981155978e-05,
        1.18632431254765e-05,
        1.2026708109602048e-05,
        1.212999908915807e-05,
        1.1755398249560798e-05,
        1.1601124527596664e-05,
        1.1614830013582949e-05,
        1.199727678433444e-05,
        1.1743226440922713e-05,
        1.1991531337139627e-05,
        1.1853352487590031e-05,
        1.1504584752912514e-05,
        1.1311027365538303e-05,
        1.1206590988838637e-05,
        1.1215846276972074e-05,
        1.196227068233118e-05,
        1.2129456635984074e-05,
        1.161745900640956e-05,
        1.1616400810283028e-05,
        1.1716157006267151e-05,
        1.1766033042072998e-05,
        1.2139652281031736e-05,
        1.2383602821955144e-05,
        1.2893132702025216e-05,
        1.2019632454098957e-05,
        1.135052687310559e-05,
        1.1215835475044279e-05,
        1.1109992687345709e-05,
        1.1615013646355464e-05,
        1.1294197569236328e-05,
        1.1187257894869849e-05,
        1.169678365056749e-05,
        1.190739001517857e-05,
        1.1509799334455872e-05,
        1.1237468201241277e-05,
        1.1286502275875175e-05,
        1.1332354298979358e-05,
        1.1265220710530857e-05,
        1.1339831197000366e-05,
        1.1786967178139693e-05,
        1.173715556110324e-05,
        1.1477661635278902e-05,
        1.1369109331291972e-05,
        1.1919456554120324e-05,
        1.1929627452933402e-05,
        1.1402445455258128e-05,
        1.1131031700311356e-05,
        1.1056610345777607e-05,
        1.1301289722229147e-05,
        1.200139663959545e-05,
        1.2200581831350853e-05,
        1.1594164943717692e-05,
        1.122227008523241e-05,
        1.1268461288869356e-05,
        1.1644597377004326e-05,
        1.169397397094867e-05,
        1.1866945044331187e-05,
        1.1903919650376068e-05,
        1.1908524806793106e-05,
        1.2216012287006656e-05,
        1.1394913958402085e-05,
        1.173604001656005e-05,
        1.2166984693529435e-05,
        1.201803710756298e-05,
        1.1255878810576376e-05,
        1.1617842376646941e-05,
        1.1478444873243383e-05,
        1.1566350765240337e-05,
        1.1276295239703658e-05,
        1.1342348046176597e-05,
        1.167566607812697e-05,
        1.1999695041369707e-05,
        1.1570537985251052e-05,
        1.1393665255548985e-05,
        1.125409865287576e-05,
        1.160697485169043e-05,
        1.2353197555597801e-05,
        1.1812109745275316e-05,
        1.136264604689551e-05,
        1.1792316881979835e-05,
        1.2073196660851388e-05,
        1.1775588231001762e-05,
        1.2000514023895255e-05,
        1.155659878482702e-05,
        1.1631805341312122e-05,
        1.1614219802861877e-05,
        1.1411108208552336e-05,
        1.2030304365963033e-05,
        1.2541662914209453e-05,
        1.1876382590446326e-05,
        1.1690780528295095e-05,
        1.1275440316218358e-05,
        1.1330560786168016e-05,
        1.1718280468872961e-05,
        1.1798457286934569e-05,
        1.1958726275223611e-05,
        1.2426471745399721e-05,
        1.236620523704853e-05,
        1.170447776553652e-05,
        1.1589376940122892e-05,
        1.1773558057772364e-05,
        1.1624446871699486e-05,
        1.1303248209937723e-05,
        1.1546190832798513e-05,
        1.1720448710379443e-05,
        1.2122036300784976e-05,
        1.2245154120609359e-05,
        1.1842598088277337e-05,
        1.150835619690246e-05,
        1.159849965914248e-05,
        1.1528407324471588e-05,
        1.160287797507582e-05,
        1.1472398543259807e-05,
        1.1442503717090449e-05,
        1.140820543595579e-05,
        1.1441550987058929e-05,
        1.1565818132000701e-05,
        1.1272643402514202e-05,
        1.114951576275546e-05,
        1.1746920503829914e-05,
        1.2578250222043215e-05,
        1.2095158944045478e-05,
        1.1444473792321568e-05,
        1.1653621307484263e-05,
        1.1869340715517427e-05,
        1.2270093022306376e-05,
        1.1841725881707563e-05,
        1.139432476234054e-05,
        1.1429410995008172e-05,
        1.153558766047495e-05,
        1.1797936830413537e-05,
        1.1334408825645965e-05,
        1.1187615536879207e-05,
        1.1680949006413468e-05,
        1.176795067885464e-05,
        1.1526269917558988e-05,
        1.1266137106805247e-05,
        1.1324986205831052e-05,
        1.1633640490645149e-05,
        1.1588289873389342e-05,
        1.12483907178302e-05,
        1.1272370801136394e-05,
        1.1508712856918384e-05,
        1.1695210693481852e-05,
        1.1894921644522839e-05,
        1.1785495169979266e-05,
        1.195366645584575e-05,
        1.190547296759299e-05,
        1.1884402923634765e-05,
        1.1754877007445018e-05,
        1.1381047032694937e-05,
        1.1448917901815114e-05,
        1.1375757623251758e-05,
        1.1676431247412229e-05,
        1.2171090604483654e-05,
        1.2183175997699377e-05,
        1.2148766554907775e-05,
        1.2053579181583559e-05,
        1.1835571139648665e-05,
        1.1755873534383777e-05,
        1.1553709367341203e-05,
        1.1260576863572448e-05,
        1.1347240926670355e-05,
        1.2080079060046293e-05,
        1.2996896609619968e-05,
        1.2137084564595522e-05,
        1.1170917131298958e-05,
        1.1463140898342125e-05,
        1.213125898673634e-05,
        1.2134448697814858e-05,
        1.1621115557167511e-05,
        1.2044533647248031e-05,
        1.1546755871821537e-05,
        1.1570222568959437e-05,
        1.1849317476561883e-05,
        1.186699218001611e-05,
        1.1594810702601148e-05,
        1.1395413400263586e-05,
        1.1784127253123045e-05,
        1.2031996929849164e-05,
        1.1663525693278835e-05,
        1.1919701463283239e-05,
        1.1866459743175161e-05,
        1.1637608333322278e-05,
        1.1479029552135122e-05,
        1.1347286098368407e-05,
        1.174685392467496e-05,
        1.1872438119212971e-05,
        1.1721668935422905e-05,
        1.1899241433647401e-05,
        1.1551801550493922e-05,
        1.1436711134210712e-05,
        1.178063842684395e-05,
        1.2252627483453996e-05,
        1.1613752373986384e-05,
        1.132326300374972e-05,
        1.1545217284506155e-05,
        1.1462922110204604e-05,
        1.1425637194233975e-05,
        1.1603967791390993e-05,
        1.1756014352242485e-05,
        1.2594014555467225e-05,
        1.2574689317445924e-05,
        1.1867179937161057e-05,
        1.161963746064778e-05,
        1.1967657112725824e-05,
        1.2314288226085491e-05,
        1.208806188108548e-05,
        1.1925569266860166e-05,
        1.1891799298194026e-05,
        1.186638452247797e-05,
        1.1717746068045139e-05,
        1.146220937936882e-05,
        1.1563748482635181e-05,
        1.1981995788079582e-05,
        1.2420658540657832e-05,
        1.2095413476744065e-05,
        1.1357990023218496e-05,
        1.1216168960015115e-05,
        1.1023294843673606e-05,
        1.1316830357548457e-05,
        1.2159771951742687e-05,
        1.2132958031779149e-05,
        1.1448022520200253e-05,
        1.1047689524608442e-05,
        1.1005648814427708e-05,
        1.1262986282666792e-05,
        1.1481524993854453e-05,
        1.1358767172823673e-05,
        1.1557004937312111e-05,
        1.1628394288913152e-05,
        1.1732053712406324e-05,
        1.174653045603717e-05,
        1.1859829127097221e-05,
        1.1439269619908629e-05,
        1.1267348297509099e-05,
        1.1589504010073498e-05,
        1.2013555093122805e-05,
        1.1902138117884645e-05,
        1.1594419672814969e-05,
        1.157793102103263e-05,
        1.1748101449135937e-05,
        1.1478122190200343e-05,
        1.1738956537064698e-05,
        1.2620112405817301e-05,
        1.2289907096660073e-05,
        1.2070419976212012e-05,
        1.1927458425832166e-05,
        1.1540474845406779e-05,
        1.1454425706598437e-05,
        1.1956187822191789e-05,
        1.197185042109584e-05,
        1.1740159675422372e-05,
        1.1777711497208883e-05,
        1.2793464492652951e-05,
        1.2667384980625869e-05,
        1.1666890199188943e-05,
        1.1699926815157145e-05,
        1.1825314611007976e-05,
        1.1482204533312102e-05,
        1.1571464201459801e-05,
        1.190067357287433e-05,
        1.1993667565660103e-05,
        1.2063494172907235e-05,
        1.2254921420120277e-05,
        1.165731065682297e-05,
        1.1562258602194221e-05,
        1.1919752723340593e-05,
        1.1976142321606821e-05,
        1.2179700722929698e-05,
        1.1603693226026313e-05,
        1.1389310900255477e-05,
        1.1675698876707725e-05,
        1.223866491158619e-05,
        1.2299344249977839e-05,
        1.1466617547902614e-05,
        1.1457643306290537e-05,
        1.2090972509629512e-05,
        1.2426305592110366e-05,
        1.1784932291341804e-05,
        1.1219249866220932e-05,
        1.1078825001284725e-05,
        1.1514971690281492e-05,
        1.2074242287461942e-05,
        1.1814739327297992e-05,
        1.1511429639958168e-05,
        1.1489665915836819e-05,
        1.1651823081104428e-05,
        1.155036607248931e-05,
        1.1273795673611898e-05,
        1.1293193971944831e-05,
        1.1947784511563347e-05,
        1.2317204550191453e-05,
        1.1528107620074946e-05,
        1.1222206255659077e-05,
        1.1540854287670415e-05,
        1.2245112680486363e-05,
        1.2272081559014088e-05,
        1.17931095470813e-05,
        1.2473881406491942e-05,
        1.25341223830138e-05,
        1.207660064289762e-05,
        1.1650882331392826e-05,
        1.162330108175847e-05,
        1.1819538525617962e-05,
        1.1744056814572117e-05,
        1.1704236784347348e-05,
        1.1539206699083648e-05,
        1.1387930806680652e-05,
        1.1416759580775989e-05,
        1.1535539542796592e-05,
        1.154197650976897e-05,
        1.142044048497115e-05,
        1.148280590609225e-05,
        1.1832041658841322e-05,
        1.1698772972869951e-05,
        1.144478037067226e-05,
        1.1283009325223651e-05,
        1.1307149277061212e-05,
        1.1780651781954679e-05,
        1.2380566105453942e-05,
        1.232633080438741e-05,
        1.1867759313288242e-05,
        1.1536337707061298e-05,
        1.1645292824755637e-05,
        1.1639381027872782e-05,
        1.1669533921917095e-05,
        1.1765357234190407e-05,
        1.2033323602981076e-05,
        1.1738804328082131e-05,
        1.170142396234953e-05,
        1.1663300620383323e-05,
        1.180558263130552e-05,
        1.1621715358758164e-05,
        1.1768895356539984e-05,
        1.1686119005454838e-05,
        1.1862159593919317e-05,
        1.1667534386882898e-05,
        1.1562592676361117e-05,
        1.1799205369534043e-05,
        1.2052829331395898e-05,
        1.214890148080587e-05,
        1.2050716474319198e-05,
        1.1774268039026527e-05,
        1.1492605611386555e-05,
        1.1523314313715595e-05,
        1.1568854848501905e-05,
        1.2036497994961994e-05,
        1.2697896105875601e-05,
        1.2534479632225784e-05,
        1.1641529236713174e-05,
        1.1260525996312469e-05,
        1.1506561505698996e-05,
        1.1851984177936435e-05,
        1.2020066495197627e-05,
        1.1831159828735876e-05,
        1.1697491078638718e-05,
        1.1736669474352465e-05,
        1.163284527236075e-05,
        1.1527546312626983e-05,
        1.198575564454699e-05,
        1.2265293823986404e-05,
        1.1805602860370299e-05,
        1.1469154036947566e-05,
        1.169145044421707e-05,
        1.2286456764523664e-05,
        1.2108965378956974e-05,
        1.2180532274971225e-05,
        1.2286543769142085e-05,
        1.2213507614549027e-05,
        1.1744620086006955e-05,
        1.1674219994593249e-05,
        1.2066547386898163e-05,
        1.2270312988836018e-05,
        1.1552302367146233e-05,
        1.1106650374487249e-05,
        1.1649223940878265e-05,
        1.2457256454020701e-05,
        1.196118145521207e-05,
        1.1318306097283939e-05,
        1.1288174021500466e-05,
        1.1532742432693749e-05,
        1.1816010026804057e-05,
        1.1831397078349992e-05,
        1.135531860827545e-05,
        1.1448619965006662e-05,
        1.1641124655417581e-05,
        1.1670402789709186e-05,
        1.197233316906893e-05,
        1.1989956612465806e-05,
        1.161303826835979e-05,
        1.1250784228630883e-05,
        1.1493588979613273e-05,
        1.2012843540679147e-05,
        1.1753983197019653e-05,
        1.1187926632399702e-05,
        1.1135026056811257e-05,
        1.1540725253732935e-05,
        1.185902703485877e-05,
        1.2053674238548154e-05,
        1.187585349238306e-05,
        1.1898108409621051e-05,
        1.1973116014236038e-05,
        1.1977378062146568e-05,
        1.1814788426969786e-05,
        1.1523639549941566e-05,
        1.1526603598928512e-05,
        1.1725855173440183e-05,
        1.2184246763341893e-05,
        1.2062416926108046e-05,
        1.1726789049197731e-05,
        1.1643066449237744e-05,
        1.1849829291540679e-05,
        1.2356832109702785e-05,
        1.2520889628667563e-05,
        1.2058995268179968e-05,
        1.181003420394918e-05,
        1.1171796801018847e-05,
        1.1416127373401952e-05,
        1.2035248702912832e-05,
        1.1753388501794868e-05,
        1.1768743540354793e-05,
        1.1693760092778328e-05,
        1.1428334141006352e-05,
        1.1713070993695466e-05,
        1.1636962574438825e-05,
        1.2479270193670833e-05,
        1.2333356181826585e-05,
        1.1247699983647381e-05,
        1.1164981373776267e-05,
        1.154939880895494e-05,
        1.2098814512809989e-05,
        1.181607385637739e-05,
        1.1282400096496013e-05,
        1.1329849233724356e-05,
        1.1920791083199725e-05,
        1.2085870660932592e-05,
        1.1839296233548438e-05,
        1.1516788574536605e-05,
        1.1389773811961165e-05,
        1.1375100080447073e-05,
        1.155974725218123e-05,
        1.1769083899279678e-05,
        1.182914359981327e-05,
        1.1656260709441296e-05,
        1.1419317673676529e-05,
        1.143862150424093e-05,
        1.1827394276706541e-05,
        1.228490305450937e-05,
        1.2001351467897398e-05,
        1.169669998472675e-05,
        1.1745971898170826e-05,
        1.1676182017478195e-05,
        1.1824506823207596e-05,
        1.1718593135582954e-05,
        1.1974059709927948e-05,
        1.2202516162020907e-05,
        1.18314995984647e-05,
        1.1580936313743883e-05,
        1.1855730090097053e-05,
        1.196595413970927e-05,
        1.1911798380111049e-05,
        1.197699390631444e-05,
        1.139973574257108e-05,
        1.1298019291290197e-05,
        1.147057890942307e-05,
        1.1804972813381822e-05,
        1.1963504066086682e-05,
        1.1907691879960767e-05,
        1.1592355326214002e-05,
        1.1457914140080158e-05,
        1.1584759803385937e-05,
        1.1543303575698255e-05,
        1.1670933458961954e-05,
        1.1840013677952714e-05,
        1.2233144537286887e-05,
        1.2091834896264926e-05,
        1.2065483888007075e-05,
        1.21380547741102e-05,
        1.1625817145339952e-05,
        1.1590000309556006e-05,
        1.1677196613096176e-05,
        1.1905625765771615e-05,
        1.2186480209212523e-05,
        1.2142457836278124e-05,
        1.1809021572318071e-05,
        1.163915949015364e-05,
        1.1628413732383181e-05,
        1.1858479475318907e-05,
        1.2027329318649604e-05,
        1.1946267528103556e-05,
        1.1958268077086416e-05,
        1.1738197652537426e-05,
        1.1426246226562924e-05,
        1.1593162132020944e-05,
        1.2069317590380863e-05,
        1.192414459078335e-05,
        1.1832684078947095e-05,
        1.1710399971549797e-05,
        1.15471027119031e-05,
        1.1729024262656545e-05,
        1.1726695170625258e-05,
        1.1706341589077875e-05,
        1.1597550267888642e-05,
        1.1674601204445069e-05,
        1.176053132564898e-05,
        1.1663141733845394e-05,
        1.1610672842571375e-05,
        1.1543954637346264e-05,
        1.1545620491010937e-05,
        1.1653539802029082e-05,
        1.170326736042742e-05,
        1.1958794229169377e-05,
        1.2207289239315482e-05,
        1.2195713107895619e-05,
        1.187264021346208e-05,
        1.1515905173241666e-05,
        1.1438891552435803e-05,
        1.1792979531150385e-05,
        1.2324432807474485e-05,
        1.2440375201262687e-05,
        1.2048108888949488e-05,
        1.1605657998492877e-05,
        1.1467096367901964e-05,
        1.1677002178395864e-05,
        1.2223037271648456e-05,
        1.1787492741026591e-05,
        1.1992101678927203e-05,
        1.215323305385166e-05,
        1.202725370515504e-05,
        1.2004349101059852e-05,
        1.212660806942519e-05,
        1.1686899690236383e-05,
        1.139563415238798e-05,
        1.1719468091734347e-05,
        1.1995803008585827e-05,
        1.179484963944973e-05,
        1.1464015461696143e-05,
        1.1435802986681186e-05,
        1.1629356446081654e-05,
        1.1595938031065571e-05,
        1.1359331229853259e-05,
        1.1385097952016746e-05,
        1.1710935943567114e-05,
        1.2001085740473641e-05,
        1.2020550814360216e-05,
        1.1769644028335522e-05,
        1.1678415463348826e-05,
        1.1675390334370165e-05,
        1.1683843333866465e-05,
        1.1492060997827e-05,
        1.1587159991741985e-05,
        1.1988544309506282e-05,
        1.1911839820234045e-05,
        1.1716144240352487e-05,
        1.1938370533290668e-05,
        1.2572505167645776e-05,
        1.2198105636702867e-05,
        1.2161737706202688e-05,
        1.1978308795525121e-05,
        1.2024496463985696e-05,
        1.2411708063286904e-05,
        1.2607990089648386e-05,
        1.2144172593215908e-05,
        1.2084154529204e-05,
        1.18582834694291e-05,
        1.216429363871767e-05,
        1.1940273832968144e-05,
        1.1955851587639334e-05,
        1.1589888558703e-05,
        1.1447987757632622e-05,
        1.167031617788814e-05,
        1.1714119762685016e-05,
        1.2060359060663756e-05,
        1.2532369524730704e-05,
        1.1825587212385785e-05,
        1.1352245950814493e-05,
        1.1534056536309681e-05,
        1.1929648074795555e-05,
        1.1880016948152623e-05,
        1.1343847942950604e-05,
        1.138611627920978e-05,
        1.164931133829406e-05,
        1.1713655279789832e-05,
        1.1428817674574194e-05,
        1.1510568038917503e-05,
        1.1628635859298384e-05,
        1.1473906099582613e-05,
        1.1304186406466388e-05,
        1.1401932461887208e-05,
        1.166189185260017e-05,
        1.1729348124091708e-05,
        1.1751468311830294e-05,
        1.2063358657813084e-05,
        1.2370002016469126e-05,
        1.1942995329576422e-05,
        1.1678464759419307e-05,
        1.1704450662517688e-05,
        1.1788333523806415e-05,
        1.2283701880138568e-05,
        1.1737471370192229e-05,
        1.1341464448482968e-05,
        1.1559691867751443e-05,
        1.240917452022226e-05,
        1.2217624916627106e-05,
        1.1642186386720483e-05,
        1.1394709114571354e-05,
        1.1813533046561321e-05,
        1.2641183824566334e-05,
        1.2480913461486482e-05,
        1.1637632490360803e-05,
        1.1442508430658941e-05,
        1.1577225360549588e-05,
        1.1859770011092377e-05,
        1.2019210196921515e-05,
        1.1801238685142436e-05,
        1.1591455231030648e-05,
        1.1460011874457945e-05,
        1.151458223168481e-05,
        1.1910262935174663e-05,
        1.191824025705061e-05,
        1.1724442281284597e-05,
        1.1722045431706232e-05,
        1.1637711639031737e-05,
        1.1573669955115538e-05,
        1.1808374438643807e-05,
        1.189443457577863e-05,
        1.1985797870264733e-05,
        1.2110977679905837e-05,
        1.1592429172120383e-05,
        1.1857123342383919e-05,
        1.2250684707640395e-05,
        1.208451904516741e-05,
        1.2027365456008046e-05,
        1.215774668848047e-05,
        1.2099942234071786e-05,
        1.2143520549574466e-05,
        1.2421384626604343e-05,
        1.2123294037977686e-05,
        1.2533400028642347e-05,
        1.2660598423990305e-05,
        1.2213463424844413e-05,
        1.2016477309189382e-05,
        1.1741896232614434e-05,
        1.2031184821277668e-05,
        1.1930065029208443e-05,
        1.1882698379428715e-05,
        1.1882010787624892e-05,
        1.172734250069821e-05,
        1.1977448569275266e-05,
        1.2143949091509897e-05,
        1.1988935732089835e-05,
        1.2029240277875883e-05,
        1.201988855798704e-05,
        1.2371206922414986e-05,
        1.2458647742320698e-05,
        1.1914569958384557e-05,
        1.164503829205705e-05,
        1.1648856086137173e-05,
        1.1811216327647326e-05,
        1.206578732397877e-05,
        1.2099845409519006e-05,
        1.204324527186012e-05,
        1.226817695671423e-05,
        1.2339500514755064e-05,
        1.2210207920205689e-05,
        1.2006089586225658e-05,
        1.1388088318427773e-05,
        1.1656807483386411e-05,
        1.2365024880938568e-05,
        1.2075461923309341e-05,
        1.1875105606182272e-05,
        1.1836024624217365e-05,
        1.2107171669746944e-05,
        1.1751812991526296e-05,
        1.1318291367382401e-05,
        1.123714198302187e-05,
        1.1431470431641958e-05,
        1.1943731628254664e-05,
        1.2060283643567878e-05,
        1.1473395659394628e-05,
        1.1272530669667761e-05,
        1.1401499991978035e-05,
        1.175399399894745e-05,
        1.1879215641508921e-05,
        1.1680600398743722e-05,
        1.1501681391119906e-05,
        1.1322930304373634e-05,
        1.1334995861321951e-05,
        1.1586946899166394e-05,
        1.1811997994422308e-05,
        1.1828344846352502e-05,
        1.1789138365626486e-05,
        1.1867665041918393e-05,
        1.1976947556224265e-05,
        1.1708442465834656e-05,
        1.1451355013124352e-05,
        1.1489799074146727e-05,
        1.1641603279018242e-05,
        1.1767811628584115e-05,
        1.1988338680080802e-05,
        1.1955627889534633e-05,
        1.1725932947320308e-05,
        1.183170935226261e-05,
        1.2207057881661981e-05,
        1.2115115996643441e-05,
        1.1884251893044322e-05,
        1.183663620972925e-05,
        1.1853424958705601e-05,
        1.2251413935965903e-05,
        1.2365583242406226e-05,
        1.2204283161009479e-05,
        1.1719886224539359e-05,
        1.1455610972675579e-05,
        1.1608507939842572e-05,
        1.1969976384822753e-05,
        1.1863043388011637e-05,
        1.1560852191195313e-05,
        1.1658775201833283e-05,
        1.1769062491822774e-05,
        1.1922547673057877e-05,
        1.2189994960118326e-05,
        1.2036386440507674e-05,
        1.1546579113003074e-05,
        1.1497713941240149e-05,
        1.1711751980112355e-05,
        1.1679987242042339e-05,
        1.1678461420641627e-05,
        1.1623269657968518e-05,
        1.1449072467581928e-05,
        1.155372704322305e-05,
        1.1740446221106971e-05
    ]
} (train.py:310)[0m
[32m2022-05-24 16:42:21,921 - util.base_logger - INFO - Total time used for train - 9276.13830113411 (decorators.py:10)[0m
