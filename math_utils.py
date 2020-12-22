import numpy as np

# math utilities class 
class MathUtils:
    # determine if a point is on a line segment 
    def point_on_segment(p1, p2, q):
        fits_x = q[0] <= max(p1[0], p2[0]) and q[0] <= min(p1[0], p2[0])
        fits_y = q[1] <= max(p1[1], p2[1]) and q[1] <= min(p1[1], p2[1])
        on_line = fits_x and fits_y
        return on_line
    
    # determine the orientation of both points of one line segement and one point of another 
    def three_point_orientation(p1, p2, q):
        res = (p2[1] - p1[1]) / 10 * (q[0] - p2[0]) / 10 - (p2[0] - p1[0]) / 10 * (q[1] - p2[1]) / 10
        if res == 0:
            return 0 # collinear
        elif res < 0: 
            return -1 # anti-clockwise
        return 1 # clockwise
    
    # determine if two lines intersect
    def lines_intersect(lines):
        line1, line2 = lines
        line1_point1 = line1[0], line1[1]
        line1_point2 = line1[2], line1[3]
        line2_point1 = line2[0], line2[1]
        line2_point2 = line2[2], line2[3]
        orientation1 = MathUtils.three_point_orientation(line1_point1, line1_point2, line2_point1)
        orientation2 = MathUtils.three_point_orientation(line1_point1, line1_point2, line2_point2)
        orientation3 = MathUtils.three_point_orientation(line2_point1, line2_point2, line1_point1)
        orientation4 = MathUtils.three_point_orientation(line2_point1, line2_point2, line1_point2)
        if orientation1 != orientation2 and orientation3 != orientation4:
            return True 
        elif orientation1 == 0 and MathUtils.point_on_segment(line1_point1, line1_point2, line2_point1):
            return True
        elif orientation2 == 0 and MathUtils.point_on_segment(line1_point1, line1_point2, line2_point2):
            return True
        elif orientation3 == 0 and MathUtils.point_on_segment(line2_point1, line2_point2, line1_point1):
            return True
        elif orientation4 == 0 and MathUtils.point_on_segment(line2_point1, line2_point2, line1_point2):
            return True
        return False
    
    # determine if one of two lines has an absolute slope less than a particular threshold
    def abs_slope_check_threshold(lines, threshold):
        line1, line2 = lines
        line1_point1 = np.array([line1[0], line1[1]])
        line1_point2 = np.array([line1[2], line1[3]])
        line1_diffs = line1_point2 - line1_point1
        slope_line1 = line1_diffs[1] / line1_diffs[0]
        line2_point1 = np.array([line2[0], line2[1]])
        line2_point2 = np.array([line2[2], line2[3]])
        line2_diffs = line2_point2 - line2_point1
        slope_line2 = line2_diffs[1] / line2_diffs[0]
        less_than_threshold = abs(slope_line1) < threshold or abs(slope_line2) < threshold
        return less_than_threshold
    
    # determine lines of a box
    def get_box_lines(box):
        x1, y1, box_width, box_height = box 
        line1 = [x1, y1, x1 + box_width, y1]
        line2 = [x1, y1, x1, y1 + box_height]
        line3 = [x1, y1 + box_height, x1 + box_width, y1 + box_height]
        line4 = [x1 + box_width, y1, x1 + box_width, y1 + box_height]
        return [line1, line2, line3, line4]
    
    # determine if one of two lines intersects a box
    def line_box_intersect(lines, box): 
        if lines is None: 
            return False
        box_lines = MathUtils.get_box_lines(box)
        for line in lines: 
            intersection_count = 0
            for box_line in box_lines: 
                if MathUtils.lines_intersect([line, box_line]):
                    intersection_count += 1
            if intersection_count > 0:
                return True
        return False