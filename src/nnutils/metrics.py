import torch


def calculate_correct_key_points(src_kp, tar_kp, tar_pred_kp, theta_x, theta_y):

    out = torch.zeros([3], dtype=torch.int32)

    source = src_kp.view(-1, 3).clone()
    target = tar_kp.view(-1, 3).clone()
    pred = tar_pred_kp.view(-1, 3).clone()

    pred[:, 2] = source[:, 2] * target[:, 2]
    target[:, 2] = source[:, 2] * target[:, 2]
    
    x, _ = torch.where(target[:, 2:] == 1)
    target = target[x, :]

    x, _ = torch.where(pred[:, 2:] == 1)
    pred = pred[x, :]

    out[2] = pred.size(0)

    pred = torch.abs(target[:, :2] - pred[:, :2])
    pred, _ = torch.where((pred[:, 0:1] < theta_x) & (pred[:, 1:] <  theta_y))
    out[0] = pred.size(0)
    out[1] = out[2] - out[0]

    return out


    



