import glob
import os
import random
import shutil
from pathlib import Path


def random_split(
    input_path: str,
    output_path: str,
    train_ratio: float,
    val_ratio: float,
    move: bool = False,
    name: str = "",
) -> None:
    """
    It takes in a path to a folder containing datas, and splits the datas into train, val, and test
    folders

    Args:
      input_path (str): The path to the folder containing the datas you want to split.
      output_path (str): The path to the folder where you want to save the train/val/test folders.
      train_ratio (float): the ratio of the training set
      val_ratio (float): float,
      name (str): str=''

    Returns:
      None
    """

    def copy_files(files, des_dir, move):
        copy_func = shutil.move if move else shutil.copy2

        for file in files:
            copy_func(file, des_dir)

    assert train_ratio + val_ratio <= 1, "Total of train, val ratio must <= 1"

    # Get data path
    fp = glob.glob(str(Path(input_path) / "**" / "*.*"), recursive=True)
    n = len(fp)
    assert n > 0, "Empty data folder"

    # Create train/val/test folder
    train_path = Path(output_path) / "train" / name
    train_path.mkdir(parents=True, exist_ok=True)
    val_path = Path(output_path) / "val" / name
    val_path.mkdir(parents=True, exist_ok=True)
    test_path = Path(output_path) / "test" / name
    if n > int(n * (train_ratio + val_ratio)):
        test_path.mkdir(parents=True, exist_ok=True)

    # Shuffle datas
    random.shuffle(fp)
    train, val, test = (
        fp[: int(n * train_ratio)],
        fp[int(n * train_ratio) : int(n * (train_ratio + val_ratio))],
        fp[int(n * (train_ratio + val_ratio)) :],
    )

    # Split
    copy_files(train, train_path, move=move)
    copy_files(val, val_path, move=move)
    if len(test) > 0:
        copy_files(test, test_path, move=move)

    print(f"Split done: train-{len(train)}, val-{len(val)}, test-{len(test)}")


def get_data_paths(
    dir: str | list[str], data_formats: list, prefix: str = ""
) -> list[str]:
    """
    It takes a directory or a list of directories and returns a list of all the files in those
    directories that have a file extension in the data_formats
    Modified from: https://github.com/ultralytics/yolov5/blob/master/utils/dataloaders.py

    Args:
      dir (str | list[str]): str | list[str]
      data_formats (list): list
      prefix (str): str = ''

    Returns:
      A list of strings.

    Example of data_formats:
      IMG_FORMATS = "bmp", "dng", "jpeg", "jpg", "png"
    """
    try:
        f = []  # data files
        for d in dir if isinstance(dir, list) else [dir]:
            p = Path(d)  # os-agnostic
            if p.is_dir():  # dir
                f += glob.glob(str(p / "**" / "*.*"), recursive=True)
            elif p.is_file():  # file
                with open(p) as t:
                    t = t.read().strip().splitlines()
                    parent = str(p.parent) + os.sep
                    f += [
                        x.replace("./", parent, 1) if x.startswith("./") else x
                        for x in t
                    ]  # to global path
            else:
                raise FileNotFoundError(f"{prefix}{p} does not exist")
        data_files = sorted(x for x in f if x.split(".")[-1].lower() in data_formats)
        assert data_files, f"{prefix}No data found"
        return data_files
    except Exception as e:
        raise Exception(f"{prefix}Error loading data from {dir}: {e}") from e
