import glob
import random
import shutil
from pathlib import Path


def random_split(
    input_path: str, 
    output_path: str, 
    train_ratio: float, 
    val_ratio: float, 
    name: str=''
) -> None:
    """
    It takes in a path to a folder containing images, and splits the images into train, val, and test
    folders
    
    Args:
      input_path (str): The path to the folder containing the images you want to split.
      output_path (str): The path to the folder where you want to save the train/val/test folders.
      train_ratio (float): the ratio of the training set
      val_ratio (float): float,
      name (str): str=''
    
    Returns:
      None
    """ 
    
    assert train_ratio + val_ratio <= 1, 'Total of train, val ratio must <= 1'

    # Get data path
    fp = glob.glob(str(Path(input_path) / '**' / '*.*'), recursive=True)
    n = len(fp)
    assert n > 0, 'Empty data folder'

    # Create train/val/test folder
    train_path = Path(output_path) / 'train' / name
    train_path.mkdir(parents=True, exist_ok=True)
    val_path = Path(output_path) / 'val' / name
    val_path.mkdir(parents=True, exist_ok=True)
    test_path = (Path(output_path) / 'test' / name)
    if n > int(n*(train_ratio+val_ratio)):
        test_path.mkdir(parents=True, exist_ok=True)

    # Shuffle images
    random.shuffle(fp)
    train, val, test = fp[:int(n*train_ratio)], fp[int(n*train_ratio):int(n*(train_ratio+val_ratio))], fp[int(n*(train_ratio+val_ratio)):]
    
    # Split
    split_train = [shutil.copy2(f, train_path) for f in train]
    split_val = [shutil.copy2(f, val_path) for f in val]
    split_test = [shutil.copy2(f, test_path) for f in test] if len(test) > 0 else None

    print(f'Split done: train-{len(train)}, val-{len(val)}, test-{len(test)}')