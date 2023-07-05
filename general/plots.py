import PIL


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


def merge_images(imgs, offset=0, axes='hor'):
    """
    Merge images in list horizontally or vertically with offset between.

    Args:
      imgs: List of images (PIL.Image).
      offset: Space (pixels) between the merged images. Defaults to 0
      axes: Orientation in which the images will be merged (hor or ver). Defaults to 'hor'.

    Returns:
      A new image that is created by merging the input images
    """
    assert axes not in ['hor', 'ver'], "axes must be 'hor' or 'ver'"

    widths, heights = zip(*(i.size for i in imgs))
    cur_pos = 0
    if axes == 'hor':
        w, h = sum(widths), max(heights)
        new_im = PIL.Image.new('RGB', (w, h))
        for im in imgs:
            new_im.paste(im, (cur_pos, 0))
            cur_pos = cur_pos + im.size[0] + offset
    elif axes == 'ver':
        w, h = max(widths), sum(heights)
        new_im = PIL.Image.new('RGB', (w, h))
        for i, im in enumerate(imgs):
            new_im.paste(im, (0, cur_pos))
            cur_pos = cur_pos + im.size[1] + offset

    return new_im
