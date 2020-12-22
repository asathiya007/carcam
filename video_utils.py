import os.path
import cv2

# extract frames from input video
def extract_frames(inputvid_filepath):
    if not os.path.isfile(inputvid_filepath): 
        print('Input file does not exist or is not accessible. Please provide a different file path or file.')
        return None
    video = cv2.VideoCapture(inputvid_filepath)
    frames = []
    valid, frame = video.read()
    while valid: 
        frames.append(frame)
        valid, frame = video.read()
    video.release()
    return frames

# generate a video from an array of frames 
def compile_frames(frames, outputvid_filepath):
    height, width, layers = frames[0].shape
    size = (width, height)
    fps = 60
    output_video = cv2.VideoWriter(outputvid_filepath, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    for frame in frames: 
        output_video.write(frame)