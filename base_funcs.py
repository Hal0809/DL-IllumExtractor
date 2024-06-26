import numpy as np
import pandas as pd
import cv2
import os
import torch
from torch import nn
from torchvision import transforms
from DatasetLoader import CustomDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

cam2rgb = np.array([
        1.8795, -1.0326, 0.1531,
        -0.2198, 1.7153, -0.4955,
        0.0069, -0.5150, 1.5081,]).reshape((3, 3))


def linearize(img, black_lvl=2048, saturation_lvl=2**14-1):
    """
    :param saturation_lvl: 2**14-1 is a common value. Not all images
                           have the same value.
    """
    return np.clip((img - black_lvl)/(saturation_lvl - black_lvl), 0, 1)


def get_preview(img_png_path, illum=(0.20072707, 0.469711179, 0.329561751)):

    illum = np.array(illum)
    illum /= illum.sum()

    cam = cv2.imread(img_png_path, cv2.IMREAD_UNCHANGED)
    cam = linearize(cv2.cvtColor(cam, cv2.COLOR_BGR2RGB).astype(np.float64))

    cam_wb = np.clip(cam/illum, 0, 1)

    rgb = np.dot(cam_wb, cam2rgb.T)
    rgb = np.clip(rgb, 0, 1)**(1/2.2)
    rgb = (rgb*255).astype(np.uint8)
    out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return out


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def Evaluation(model, model_path, result_csv, batch_size=1, batch_limit=50):
    """
        调用模型，读取图片，进行光照估计，保存结果。结果包括测试集上的50张图片的光照估计值（rgb）和真实值，以及误差。
        :param model: 要调用的神经网络
        :param model_path: 训练权重保存的路径
        :param result_csv: 光照估计结果存放文件名，以 eval_ 开头
        :param batch_size: 一轮评估的图片数量
        :param batch_limit: 要评估的图片数量总数
    """
    # 定义测试集
    test_data_dir = os.path.join('data', 'SimpleCube++', 'test', 'PNG')
    test_csv_file = os.path.join('data', 'SimpleCube++', 'test', 'gt.csv')
    # 创建测试集实例
    test_dataset = CustomDataset(test_data_dir, test_csv_file, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 定义网络模型，加载参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model
    model.load_state_dict(torch.load(model_path, map_location=device))  # '../models/Den_CBAM.pth'

    model.eval()
    model.to(device)
    criterion = nn.MSELoss()

    # 定义结果保存方式
    outputs = []
    with torch.no_grad():
        for batch_ind, (inputs, targets) in enumerate(test_loader):
            print(batch_ind+1)
            inputs, targets = inputs.to(device), targets.to(device)
            pred = model(inputs)
            mse = criterion(pred, targets)

            illumination = torch.cat([pred[:, 0:1], pred[:, 1:2], 1.0-pred[:, 0:1]-pred[:, 1:2], targets[:, 0:1],
                                      targets[:, 1:2], 1.0-targets[:, 0:1]-targets[:, 1:2],
                                      mse.unsqueeze(0).unsqueeze(1)], dim=1)

            outputs.append(illumination.cpu().numpy())
            if batch_ind + 1 == batch_limit:
                break

    outputs = np.concatenate(outputs, axis=0)
    print(outputs)
    # 读取图片名称，并加入结果第一列中
    img_names = np.array(pd.read_csv(test_csv_file).iloc[:batch_limit, 0].values)
    img_names = np.reshape(img_names, (-1, 1))
    outputs = np.concatenate((img_names, outputs), axis=1)
    csvout = pd.DataFrame(data=outputs, index=None, columns=['img_names', 'pred_r', 'pred_g', 'pred_b', 'gt_r', 'gt_g',
                                                             'gt_b', 'mse_loss'])
    csvout.to_csv(os.path.join('results', result_csv))  # 保存结果

    return


def find_top_low5(csv_file, result_file, model_name):
    """
    找到5个最好和最差的结果，并且根据最好的5个结果展示图片
    :param csv_file: Evaluation()保存的结果，由 eval_ 开头
    :param result_file: 保存结果，由 res_ 开头
    :param model_name: 特征提取使用的模型
    :return: None
    """
    csv_file = pd.read_csv(os.path.join('results', csv_file))
    csv_file['mse_loss'] = pd.to_numeric(csv_file['mse_loss'], errors='coerce')  # 要改一下数据格式

    top_5 = csv_file.nsmallest(5, 'mse_loss')
    illums = top_5[['img_names', 'pred_r', 'pred_g', 'pred_b', 'gt_r', 'gt_g', 'gt_b']].values.tolist()

    for illum in illums:
        print(illum[0])
        _img = get_preview(os.path.join('data', 'SimpleCube++', 'test', 'PNG', illum[0] + '.png'), illum[1:4])  # 还原
        cv2.imwrite(os.path.join('results', 'test_top5', model_name, illum[0] + '.png'), _img)  # 保存5张示例图片
        gt_img = get_preview(os.path.join('data', 'SimpleCube++', 'test', 'PNG', illum[0] + '.png'), illum[4:7])  # 标准图像
        cv2.imwrite(os.path.join('results', 'test_top5', model_name, illum[0] + '_gt.png'), gt_img)

    low5 = csv_file.nlargest(5, 'mse_loss')
    merged_df = pd.concat([top_5, low5], axis=0)
    merged_df.to_csv(os.path.join('results', result_file))


def DrawResults(csv_file, model_idx):
    csv_file = pd.read_csv(os.path.join('results', csv_file))
    csv_file['mse_loss'] = pd.to_numeric(csv_file['mse_loss'], errors='coerce')  # 要改一下数据格式

    top_5 = csv_file.nsmallest(5, 'mse_loss')
    img_names = top_5['img_names'].values.tolist()

    img_batches = []
    gt_img_names = []
    for img_name in img_names:
        img = mpimg.imread(os.path.join('results', 'test_top5', model_idx, img_name+'.png'))
        img_batches.append(img)
        gt_img_names.append(img_name+'_gt')

    for gt_img_name in gt_img_names:
        img = mpimg.imread(os.path.join('results', 'test_top5', model_idx, gt_img_name+'.png'))
        img_batches.append(img)

    fig, axs = plt.subplots(2, 5)

    for idx in range(5):
        axs[0, idx].imshow(img_batches[idx])
        axs[0, idx].set_title(img_names[idx])
        axs[0, idx].set_axis_off()

        axs[1, idx].imshow(img_batches[idx+5])
        axs[1, idx].set_title(gt_img_names[idx])
        axs[1, idx].set_axis_off()

    plt.tight_layout()
    fig.tight_layout()
    plt.show()
    fig.savefig(os.path.join('results', 'test_top5', model_idx, 'copm_res.png'))
