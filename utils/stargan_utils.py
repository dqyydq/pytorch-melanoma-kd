import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
import torch.autograd # 需要导入 autograd 来使用 torch.autograd.grad

# --- 网络工具 ---

def print_network(network, name):
    """打印网络的参数数量"""
    num_params = 0
    for p in network.parameters():
        num_params += p.numel()
    # print(network) # 取消注释可以打印网络结构
    print("Number of parameters of %s: %i" % (name, num_params))

def he_init(module):
    """应用 He (Kaiming Normal) 初始化"""
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)): # 同时考虑转置卷积
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

# --- 图像处理 ---

def denormalize(x):
    """将 [-1, 1] 范围的张量反归一化到 [0, 1]"""
    out = (x + 1) / 2
    return out.clamp_(0, 1) # clamp_ 确保值在 [0, 1] 区间内

def save_image(x, ncol, filename):
    """将一批图像张量保存为网格图"""
    x = denormalize(x) # 先反归一化
    vutils.save_image(x.cpu(), filename, nrow=ncol, padding=0)

# --- 训练辅助 ---

def moving_average(model, model_test, beta=0.999):
    """计算模型的指数移动平均 (EMA)"""
    for param, param_test in zip(model.parameters(), model_test.parameters()):
        # 使用 lerp 进行线性插值: param_test = beta * param_test + (1 - beta) * param
        param_test.data = torch.lerp(param.data, param_test.data, beta)

def adv_loss(logits, target):
    """计算对抗性损失 (BCEWithLogitsLoss)"""
    assert target in [1, 0]
    targets = torch.full_like(logits, fill_value=target)
    # 使用 PyTorch 内置函数，效率更高
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss

def r1_reg(d_out, x_in):
    """计算 R1 正则化项 (梯度惩罚)"""
    # zero-centered gradient penalty for real images
    batch_size = x_in.size(0)
    # 计算判别器输出相对于输入的梯度
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True,  # 需要创建计算图以便进行可能的二阶梯度计算 (虽然这里没用到)
        retain_graph=True,  # 需要保留计算图，因为后续可能还有其他操作依赖它
        only_inputs=True    # 只计算相对于 inputs 的梯度
    )[0]
    # 计算梯度的平方
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    # 计算正则化项: 0.5 * mean(sum(grad_square))
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg