# pretraing_LayoutLMv3_PubLayNet

PubLayNet:<br/>
https://github.com/ibm-aur-nlp/PubLayNet<br/>
上記URLからPDFのデータセットをダウンロード

PDFからiamge(png)を生成
```
 sh ./scirpt/pdf2image.sh
 ```
新しい辞書を作成
```
 sh ./script/create_vocab.sh
 ```
前処理
```
 sh ./script/preprocessing.sh
```
学習
```
 sh ./script/pretrain.sh
```
