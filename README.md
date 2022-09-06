
PubLayNet:\n
https://github.com/ibm-aur-nlp/PubLayNet\n
上記URLからPDFのデータセットをダウンロード\n

PDFからiamge(png)を生成\n
 sh ./scirpt/pdf2image.sh\n
 
新しい辞書を作成\n
 sh ./script/create_vocab.sh\n
 
前処理\n
 sh ./script/preprocessing.sh\n

学習\n
 sh ./script/pretrain.sh\n
