import torch


def get_gt_positions_grid(img_size):

    x = torch.linspace(-1, 1, img_size[1]).view(1, -1).repeat(img_size[0], 1)
    y = torch.linspace(-1, 1, img_size[0]).view(-1, 1).repeat(1, img_size[1])
    grid = torch.cat((x.unsqueeze(2), y.unsqueeze(2)), 2)
    grid.unsqueeze(0)

    return grid