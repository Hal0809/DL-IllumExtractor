"""草稿纸"""
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import os


def DrawResults(csv_file, model_idx):
    csv_file = pd.read_csv(os.path.join('results', csv_file))
    csv_file['mse_loss'] = pd.to_numeric(csv_file['mse_loss'], errors='coerce')  # 要改一下数据格式

    top_5 = csv_file.nsmallest(5, 'mse_loss')
    img_names = top_5['img_names'].values.tolist()
    for img_name in img_names:
        print(img_name.dtype)


if __name__ == '__main__':
    DrawResults(csv_file='eval_den.csv', model_idx=0)
