"""Read ordinary RGB/grayscale photos and downsample before stacking."""
import warnings
import numpy as np
from PIL import Image, ImageOps


class CorruptedImageError(ValueError):
    pass


def load_image(path, size):
    try:
        with warnings.catch_warnings(), Image.open(path) as image:
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            image = ImageOps.exif_transpose(image)
            if image.mode == "P" and isinstance(image.info.get("transparency"), bytes):
                image = image.convert("RGBA")
            rgb = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
            return np.asarray(rgb)
    except (OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise CorruptedImageError(f"{path}: {exc}") from exc
