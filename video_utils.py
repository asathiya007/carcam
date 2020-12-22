import cv2

# video utilities class 
class VideoUtils: 
    # extract frames from input video
    def extract_frames(inputvid_filename):
        video = cv2.VideoCapture(inputvid_filename)
        frames = []
        valid, frame = video.read()
        while valid: 
            frames.append(frame)
            valid, frame = video.read()
        video.release()
        return frames
    
    # generate a video from an array of frames 
    def compile_frames(frames, outputvid_filename):
        height, width, layers = frames[0].shape
        size = (width, height)
        fps = 60
        output_video = cv2.VideoWriter(outputvid_filename, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
        for frame in frames: 
            output_video.write(frame)