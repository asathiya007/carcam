# import code
import sys 
sys.path.append('./src')
from video_utils import *
from lane_detection import detect_lanes
from road_entity_detection import detect_road_entities

# CarCam computer vision pipeline
def carcam_pipeline(inputvid_filepath, outputvid_filepath):
    print('Input video: {}'.format(inputvid_filepath))
    print('Executing CarCam computer vision pipeline. Please wait...')
    frames = extract_frames(inputvid_filepath)
    if frames is None: 
        return
    processed_frames, lane_lines = detect_lanes(frames)
    processed_frames = detect_road_entities(processed_frames, lane_lines)
    compile_frames(processed_frames, outputvid_filepath)
    cv2.destroyAllWindows()
    print('Output video: {}'.format(outputvid_filepath))
    print('All done, check out your output video!')

# execute pipeline
if __name__ == '__main__':
    carcam_pipeline('input.mov', 'output.mp4')
