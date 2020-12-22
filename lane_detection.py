import numpy as np
import cv2
from math_utils import lines_intersect, abs_slope_check_threshold

# lane detection class 
class LaneDetection: 
    # convert a color frame to a grayscale one
    def _grayscale_frame(frame):
        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return grayscale_frame
    
    # mask a frame to only include a region of interest
    def _mask_frame(frame):
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]
        polygons = np.array([ 
            [(frame_width * 3 // 11, frame_height), (frame_width * 9 // 11, frame_height), (frame_width * 5 // 8, frame_height // 5 * 4), (frame_width * 3 // 7, frame_height // 5 * 4)] 
        ])
        mask = np.zeros_like(frame) 
        cv2.fillPoly(mask, polygons, 255)
        masked_frame = cv2.bitwise_and(frame, mask)
        return masked_frame
    
    # threshold frame to emphasize lane lines 
    def _threshold_frame(frame):
        ret, thresholded_frame = cv2.threshold(frame, 40, 145, cv2.THRESH_BINARY)
        return thresholded_frame
    
    # determine start and end points for lane marker
    def _mark_lane_points(frame, params):
        slope, intercept = params[0], params[1]
        y1 = frame.shape[0]
        y2 = int(y1 * 4 / 5)
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return np.array([x1, y1, x2, y2])
    
    # find both lane lines and and their marker points
    def _find_lane_lines(frame, lines):
        left_lane_fit, right_lane_fit = [], []
        if lines is None: 
            return None
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope, intercept = parameters[0], parameters[1]
            if slope < 0: 
                left_lane_fit.append((slope, intercept))
            else: 
                right_lane_fit.append((slope, intercept))
        if len(left_lane_fit) == 0:
            return None
        left_lane_avgfit = np.average(left_lane_fit, axis=0)
        # if np.any(np.isnan(left_lane_avgfit)): 
        #     return None
        left_lane_line = LaneDetection._mark_lane_points(frame, left_lane_avgfit)
        if len(right_lane_fit) == 0:
            return None
        right_lane_avgfit = np.average(right_lane_fit, axis=0)
        # if np.any(np.isnan(right_lane_avgfit)): 
        #     return None
        right_lane_line = LaneDetection._mark_lane_points(frame, right_lane_avgfit)
        return np.array([left_lane_line, right_lane_line])
    
    # draw lane lines on the image
    def _draw_lane_lines(frame, lane_lines): 
        lane_line_frame = np.zeros_like(frame)
        if lane_lines is not None: 
            for x1, y1, x2, y2 in lane_lines:
                try: 
                    cv2.line(lane_line_frame, (x1, y1), (x2, y2), (252, 173, 76), 10)
                except: 
                    return None
        return lane_line_frame
    
    # return an array of frames with lanes detected
    def detect_lanes(frames):
        lane_detected_frames = []
        lane_lines_per_frame = []
        for frame in frames: 
            grayscaled_frame = LaneDetection._grayscale_frame(frame)
            masked_frame = LaneDetection._mask_frame(grayscaled_frame)
            thresholded_frame = LaneDetection._threshold_frame(masked_frame)
            detected_lines = cv2.HoughLinesP(thresholded_frame, 2, np.pi / 180, 100, np.array([]), minLineLength = 10, maxLineGap = 5)
            lane_lines = LaneDetection._find_lane_lines(frame, detected_lines)
            valid_lane_lines = list(filter(None.__ne__, lane_lines_per_frame))
            if lane_lines is None:
                if len(valid_lane_lines) >= 1:
                    lane_lines = valid_lane_lines[-1]
                else:
                    lane_detected_frames.append(frame)
                    lane_lines_per_frame.append(None)
                    continue
            elif lines_intersect(lane_lines) or abs_slope_check_threshold(lane_lines, 0.6):
                if len(valid_lane_lines) >= 1:
                    lane_lines = valid_lane_lines[-1]
            annotated_frame = LaneDetection._draw_lane_lines(frame, lane_lines)
            if annotated_frame is None: 
                lane_detected_frames.append(frame)
                lane_lines_per_frame.append(None)
                continue
            lane_lines_per_frame.append(lane_lines)
            combined_frame = cv2.addWeighted(frame, 0.8, annotated_frame, 1, 1)
            lane_detected_frames.append(combined_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        return lane_detected_frames, lane_lines_per_frame