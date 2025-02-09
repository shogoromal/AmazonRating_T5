# Amazon レビュー評価予測モデル

このプロジェクトは、Amazonのレビューテキストから評価（1-5星）を予測するT5モデルを実装したものです。

## 実行環境

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shogoromal/AmazonRating_T5/blob/master/Notebooks/RatingAmazonReview_train.ipynb)

google colab の T4 で学習可能です。
notebook は RatingAmazonReview_train.ipynb を実行してください。

## 主なファイル

### データ前処理（utils.py）

- `rating_num_check()`: レビュー評価の分布を可視化
- `extract_and_format_samples()`: データセットの前処理とフォーマット

### モデルの定義 (models.py)

- `T5Classifier`: T5モデルの定義
- `tokenize_function_inputs()`: データセットのトークン化
- `train()`: モデルのトレーニング
- `get_labels()`: ラベルの取得
- `get_metrics()`: モデルの評価

### 実行（RatingAmazonReview_train.ipynb）

- モデルのトレーニング・評価
- モデルの保存・ロード・推論

## 参考
- 元論文
https://arxiv.org/abs/2302.08624
- データセット
https://amazon-reviews-2023.github.io/