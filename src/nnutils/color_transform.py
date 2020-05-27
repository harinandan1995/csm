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


def sample_uv_contour(img, uv_map, uv_texture_map, mask, real_img=True):
    """
    Converts UV values to colorful image using the texture map
    Args
        img : torch.FloatTensor  B X C X H x W  -- this is the image
        uv_map: torch.FloatTensor B X H x W x 2 -- uv predictions
        uv_img: torch.FloatTensor C x H x W  -- this the texture map
        mask: torch.FloatTensor  B X 1 x  H x W -- mask for the object
    Returns
        uv_rendering: torch.FloatTensor  C x H x W
    """

    uv_texture_map = uv_texture_map.unsqueeze(0).repeat(img.size(0), 1, 1, 1)

    uv_sample = torch.nn.functional.grid_sample(uv_texture_map, 2 * uv_map - 1)
    uv_sample = uv_sample * mask + (1 - mask)
    alphas = torch.ones_like(uv_texture_map)

    alpha_sample = torch.nn.functional.grid_sample(alphas, 2 * uv_map - 1)
    alpha_sample = (alpha_sample > 0.0).float() * 0.7
    alpha_sample = alpha_sample * mask

    if real_img:
        uv_rendering = (uv_sample * alpha_sample) * 1.0 + img * (1 - alpha_sample) * 0.4 * (
            mask) + img * (1 - alpha_sample) * (1 - mask)
    else:
        uv_rendering = (uv_sample * alpha_sample) + (img * (1 - alpha_sample))

    return uv_sample, uv_rendering
