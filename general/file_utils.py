from pathlib import Path


def increment_path(path: str, exist_ok: bool = False, sep: str = '') -> Path:
    """
    Increments a path by adding a number to the end if it already exists.

    Args:
      path (str): Path to increment.
      exist_ok (bool): If True, the path will not be incremented and returned as-is.
      sep (str): Separator to use between the path and the incrementation number.

    Returns:
      Incremented path.
    """
    path = Path(path)
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')

        for n in range(1, 999):
            p = f'{path}{sep}{n}{suffix}'
            if not Path(p).exists():
                path = Path(p)
                break

    return path
