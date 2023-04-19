def get_palette(num_cls: int) -> list[int]:
    """
    Returns the color map for visualizing the segmentation mask.

    Args:
      num_cls: Number of classes or categories for which a color palette is needed.

    Returns:
      RGB color map.

    To use this palette: PIL.Image.putpalette(get_palette(num_cls))
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= ((lab >> 0) & 1) << (7 - i)
            palette[j * 3 + 1] |= ((lab >> 1) & 1) << (7 - i)
            palette[j * 3 + 2] |= ((lab >> 2) & 1) << (7 - i)
            i += 1
            lab >>= 3
    return palette
