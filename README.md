# pretraing_LayoutLMv3_PubLayNet

## 利用方法
### 0. Download dataset
PubLayNet:<br/>
https://github.com/ibm-aur-nlp/PubLayNet<br/>
上記URLからPDFのデータセットをダウンロード

### 1. create image(src/pdf2image.py)
PDFからiamge(png)を生成
```
 sh ./scirpt/pdf2image.sh
 ```
### 2. create vocab for tokenizer (src/create_vocab.py)
新しい辞書を作成
```
 sh ./script/create_vocab.sh
 ```
### 3. preprocessing (src/preprocessing.py)
前処理
```
 sh ./script/preprocessing.sh
```
### 4. train (src/pretrain_3.py)
学習
```
 sh ./script/pretrain.sh
```

・MIMのみ作成

## Done
・元コードのLayoutLMv3はfine tuning用に書かれたものなので、事前学習用にMasked language model(MLM)を作成<br/>
・LayoutLMv3は[span mask](https://aclanthology.org/2020.tacl-1.5/)を使用<br/>
・↑実装コードがなかったため自分で実装 (src/utils/utils.py)<br/>
・論文の文字数は長いため512で切り捨てずに512ごとに分割(src/utils/utils.py)<br/>
・Masked image model(MIM)の実装<br/>
・Word-Patch Alignment (WPA)の実装<br/>

## error
・src/pretrain_3.py
```
・Traceback (most recent call last):
  File "./src/pretrain_3.py", line 360, in <module>
  File "./src/pretrain_3.py", line 325, in main
  File "./src/pretrain_3.py", line 90, in save_loss_epcoh
  File "./src/pretrain_3.py", line 58, in plot_graph
  File "/home/is/hikaru-si/.pyenv/versions/exp_005/lib/python3.8/site-packages/matplotlib/figure.py", line 3058, in savefig
    self.canvas.print_figure(fname, **kwargs)
  File "/home/is/hikaru-si/.pyenv/versions/exp_005/lib/python3.8/site-packages/matplotlib/backend_bases.py", line 2319, in print_figure
    result = print_method(
  File "/home/is/hikaru-si/.pyenv/versions/exp_005/lib/python3.8/site-packages/matplotlib/backend_bases.py", line 1648, in wrapper
    return func(*args, **kwargs)
  File "/home/is/hikaru-si/.pyenv/versions/exp_005/lib/python3.8/site-packages/matplotlib/_api/deprecation.py", line 415, in wrapper
    return func(*inner_args, **inner_kwargs)
  File "/home/is/hikaru-si/.pyenv/versions/exp_005/lib/python3.8/site-packages/matplotlib/backends/backend_agg.py", line 541, in print_png
    mpl.image.imsave(
  File "/home/is/hikaru-si/.pyenv/versions/exp_005/lib/python3.8/site-packages/matplotlib/image.py", line 1675, in imsave
    image.save(fname, **pil_kwargs)
  File "/home/is/hikaru-si/.pyenv/versions/exp_005/lib/python3.8/site-packages/PIL/Image.py", line 2317, in save
    fp = builtins.open(filename, "w+b")
PermissionError: [Errno 13] Permission denied: '/cl/work2/hikaru-si/development/exp_005/data/train/pretrain_lr_1e-4_dtasize_18_batch32/epoch_17/loss.png'
pyenv: cannot change working directory to `/project/cl-work2/hikaru-si/development/exp_005'
pyenv: cannot change working directory to `/project/cl-work2/hikaru-si/development/exp_005'
```
