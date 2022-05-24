Traceback (most recent call last):
  File "main.py", line 61, in <module>
    result, config = train.train(config)
  File "/scicore/home/dokman0000/jabjan00/cvae-papyri/util/decorators.py", line 8, in wrapper
    result = func(*args, **kwargs)
  File "/scicore/home/dokman0000/jabjan00/cvae-papyri/autoencoders/train.py", line 60, in train
    util.report.header1("Auto-Encoder")
  File "/scicore/home/dokman0000/jabjan00/cvae-papyri/util/report.py", line 46, in header1
    mdFile.new_header(level=1, title=headertext)
AttributeError: 'NoneType' object has no attribute 'new_header'
