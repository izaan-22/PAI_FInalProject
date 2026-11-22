"""Vector operations across N images, their foreground pixels and RGB channels."""
import numpy as np
import pandas as pd
from matplotlib.colors import rgb_to_hsv


def preprocess(images, gain):
    rgb = np.clip(images.astype("float32") / 255 * np.asarray(gain, dtype="float32"), 0, 1)
    return rgb, rgb_to_hsv(rgb), rgb @ np.array([.299, .587, .114], dtype="float32")


def features(rgb, hsv, gray):
    """Estimate foreground from border color; dark or brown pixels proxy surface defects."""
    edge = np.concatenate((rgb[:, 0], rgb[:, -1], rgb[:, :, 0], rgb[:, :, -1]), axis=1)
    background = np.median(edge, axis=1)[:, None, None, :]
    mask = np.linalg.norm(rgb - background, axis=-1) > .15
    mask |= (mask.mean((1, 2)) < .03)[:, None, None]
    count = mask.sum((1, 2))
    h, s, v = hsv.transpose(3, 0, 1, 2)
    brown = (h < .14) & (s > .28) & (v < .58)
    defects = mask & ((gray < .23) | brown)
    flat = rgb.reshape(len(rgb), -1, 3)
    bins = np.minimum((flat * 8).astype(int), 7)
    offsets = 8 * (np.arange(len(rgb))[:, None, None] * 3 + np.arange(3))
    votes = np.broadcast_to(mask.reshape(len(rgb), -1, 1), bins.shape)
    hist = np.bincount((bins + offsets)[votes], minlength=len(rgb)*24).reshape(-1, 24)
    hist = hist / count[:, None]
    means = (rgb * mask[..., None]).sum((1, 2)) / count[:, None]
    gray_mean = (gray * mask).sum((1, 2)) / count
    # Circular hue deviations keep red samples around 0/1 close together.
    center = np.angle((np.exp(2j*np.pi*h) * mask).sum((1, 2))) / (2*np.pi)
    delta = (h - center[:, None, None] + .5) % 1 - .5
    low, high = np.nanpercentile(np.where(mask, delta, np.nan), [10, 90], axis=(1, 2))
    stats = pd.DataFrame(means, columns=["mean_red", "mean_green", "mean_blue"])
    stats["gray_mean"] = gray_mean
    stats["foreground_ratio"] = mask.mean((1, 2))
    stats["defect_ratio"] = defects.sum((1, 2)) / count
    stats["hue_spread"] = high - low
    columns = [f"{c}_bin{i}" for c in "rgb" for i in range(8)]
    return pd.concat([stats, pd.DataFrame(hist, columns=columns)], axis=1), np.stack([low, high], axis=1)
