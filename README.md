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
### 4. train (src/pretrain_2.py)
学習
```
 sh ./script/pretrain.sh
```

・MIMのみ作成

## Do
・元コードのLayoutLMv3はfine tuning用に書かれたものなので、事前学習用にMasked langage model(MIM)を作成
・LayoutLMv3は[span mask](https://aclanthology.org/2020.tacl-1.5/)を使用
・実装コードがなかったため自分で実装 (src/utils/utils.py)
・
