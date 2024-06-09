import tqdm
import matplotlib.pyplot as plt
from datasets import concatenate_datasets, DatasetDict

def rating_num_check(dataset):
  count_rating = [0] * 5
  #for i in tqdm.tqdm(dataset):
  for i in dataset:
    count_rating[int(i['rating'])-1] += 1
  print(count_rating)

  # 各バーにラベルをつける
  labels = ['1', '2', '3', '4', '5']

  # 棒グラフを作成
  plt.figure(figsize=(10, 5))
  plt.bar(labels, count_rating, color='blue')

  # タイトルと軸ラベルを追加
  plt.title('Amazon Review Stars')
  plt.xlabel('Stars')
  plt.ylabel('Quantity')

  # グラフを表示
  plt.show()
  

def extract_and_format_samples(dataset, rating_sentence, max_samples_per_rating, ex_samples_per_rating, seed):
    
    # 各レーティング値に対してデータを抽出
    ratings_range = range(1, len(rating_sentence) + 1)
    temp_datasets = []
    formatted_text = []  # フォーマットされたテキストを格納するリスト

    definition_text = \
"""
Definition : The output will be 5 rank rating (output),
which is Very Poor, Poor, Average, Good or Excellent.
These rating is belong the sentence (input) to explain the rating.
"""

    formatted_text.append(definition_text)

    for rating in ratings_range:
        # レーティングに基づいてデータを抽出し、シャッフルして最初の1000個を選択
        filtered_dataset = dataset.filter(lambda x: x['rating'] == float(rating))
        selected_dataset = filtered_dataset.shuffle(seed).select(range(min(max_samples_per_rating, len(filtered_dataset))))
        temp_datasets.append(selected_dataset)

    # 1000個のデータを抽出したデータセット
    combined_dataset = concatenate_datasets(temp_datasets)

    # 各レーティングからさらに2個のサンプルを抽出し、フォーマットされたテキストを作成
    for rating in ratings_range:
        sub_dataset = combined_dataset.filter(lambda x: x['rating'] == float(rating)).shuffle(seed=1).select(range(ex_samples_per_rating))

        for j in range(ex_samples_per_rating):
            formatted_text.append(f"example{rating}:\n")#サンプル番号識別ようの記述 ###-example-{j+1}
            formatted_text.append(f"input: {sub_dataset[j]['text']}\n")
            formatted_text.append(f"output: {rating_sentence[rating]}\n")
    
    formatted_text.append("Now complete the following input-\ninput:")
    bos_instruction = "".join(formatted_text)
    
    #データセットのtextに指示文の内容を追加する
    def update_text(example):
        # テキストの前後に追加する内容
        suffix_text = "\noutput:"
        # textフィールドを更新
        example['labels'] = rating_sentence[int(example['rating'])]
        example['instruct_text'] = bos_instruction + example['text'] + suffix_text
        return example
    
    combined_dataset = combined_dataset.map(update_text)

    return bos_instruction, combined_dataset  # リストの中身を空白で連結して返す

