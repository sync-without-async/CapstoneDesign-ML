import skvideo.io as skvideo
import numpy as np

def webm_to_array(filename):
    """Converts a webm video file into a numpy array.

    Args:
        filename (str): The name of the webm file.

    Returns:
        numpy array: A numpy array of the video file.
    """
    video = skvideo.vread(filename)
    return video

def main():
    """Main function for testing purposes."""
    video_path = "videos/webm/ec07c4c7eb818d6c.webm"
    video = webm_to_array(video_path)
    print(video.shape)