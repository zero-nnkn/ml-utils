import cv2


def frames2video(
    img_paths: list[str],
    video_path: str,
    fourcc_code: str = "mp4v",
    fps: float = 30,
    size: tuple[int, int] | None = None,
) -> None:
    """
    Generates a video from a list of frame paths using OpenCV.

    Args:
      img_paths (list[str]): List of frame paths.
      video_path (str): The path where the generated video will be saved.
      fourcc_code (str): 4-character code of codec used to compress the frames. Defaults to "mp4v".
      fps (float): FPS of the output video. Defaults to 30.
      size (tuple[int, int]): Size (w, h) of output video. If None, use (w, h) of first frame.
    """
    if size:
        w, h = size
    else:
        h, w, _ = cv2.imread(img_paths[0]).shape

    video = cv2.VideoWriter(
        video_path, cv2.VideoWriter_fourcc(*fourcc_code), fps, (w, h)
    )

    for p in img_paths:
        video.write(cv2.imread(p))

    cv2.destroyAllWindows()
    video.release()
