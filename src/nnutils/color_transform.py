import torch


def uv_to_rgb(uv, max_value=None, device='cuda:0'):
    # TODO: Add documentation

    b, _, h, w = uv.shape

    uv_np = uv.detach()

    if max_value is not None:
        normalized_uv = uv_np / max_value
    else:
        normalized_uv = uv_np / (torch.abs(uv_np).max())

    rgb_map = torch.ones((b, 3, h, w), dtype=torch.float).to(device)
    rgb_map[:, 0] += normalized_uv[:, 0]
    rgb_map[:, 1] -= 0.5 * (normalized_uv[:, 0] + normalized_uv[:, 1])
    rgb_map[:, 2] += normalized_uv[:, 1]

    return torch.mul(rgb_map.clamp(0, 1), 255).to(torch.long)
