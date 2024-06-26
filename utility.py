import os.path

import base_funcs
from arch.densenet201_regression import DenseNetRegression
from arch.densenet201_CBAM import DenseNet_CBAM
from arch.densenet201_RVFL import DenseNet_RVFL
from arch.densenet201_CBAM_RVFL import DenseNet_CBAM_RVFL

"""
    den_开头是训练结果，包含训练时训练集和测试集的损失率
    eval_开头是评估结果，用训练好的模型选取若干（50）图片进行评估，得到估计值，并与真实值比较
    res_开头是top5和low5
"""

model1 = DenseNetRegression(pretrained=False)
model2 = DenseNet_CBAM(pretrained=False)
model3 = DenseNet_RVFL(pretrained=False)
model4 = DenseNet_CBAM_RVFL(pretrained=False)
models = [model1, model2, model3, model4]

model1_path = os.path.join('models', 'Den_.pth')
model2_path = os.path.join('models', 'Den_CBAM.pth')
model3_path = os.path.join('models', 'Den_RVFL.pth')
model4_path = os.path.join('models', 'Den_CBAM_RVFL.pth')
model_paths = [model1_path, model2_path, model3_path, model4_path]

model1_eval = 'eval_den.csv'
model2_eval = 'eval_den_CBAM.csv'
model3_eval = 'eval_den_RVFL.csv'
model4_eval = 'eval_den_CBAM_RVFL.csv'
model_evals = [model1_eval, model2_eval, model3_eval, model4_eval]

model1_res = 'res_den.csv'
model2_res = 'res_den_CBAM.csv'
model3_res = 'res_den_RVFL.csv'
model4_res = 'res_den_CBAM_RVFL.csv'
model_results = [model1_res, model2_res, model3_res, model4_res]

for i, model in enumerate(models):
    print(f'{i+1}')
    if i == 2:
        continue
    base_funcs.Evaluation(model, model_paths[i], model_evals[i])
    base_funcs.find_top_low5(model_evals[i], model_results[i], f'{i}')
    base_funcs.DrawResults(model_evals[i], f'{i}')

print('Task Finished!')
