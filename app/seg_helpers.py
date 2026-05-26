
from constants import * 
import numpy as np 
class CanvasMapper():
    @staticmethod
    def get_canvas_mapping(class_name, side_hint=None):
        side_hint = side_hint.lower() if side_hint is not None else None
        if class_name == "Half Central Circle":
            if side_hint == "left":
                return np.array([
                    [CENTER_X - CENTER_CIRCLE_RADIUS, CENTER_Y - CENTER_CIRCLE_RADIUS],
                    [CENTER_X, CENTER_Y - CENTER_CIRCLE_RADIUS],
                    [CENTER_X, CENTER_Y + CENTER_CIRCLE_RADIUS],
                    [CENTER_X - CENTER_CIRCLE_RADIUS, CENTER_Y + CENTER_CIRCLE_RADIUS],
                ], dtype=np.float32)

            if side_hint == "right":
                return np.array([
                    [CENTER_X, CENTER_Y - CENTER_CIRCLE_RADIUS],
                    [CENTER_X + CENTER_CIRCLE_RADIUS, CENTER_Y - CENTER_CIRCLE_RADIUS],
                    [CENTER_X + CENTER_CIRCLE_RADIUS, CENTER_Y + CENTER_CIRCLE_RADIUS],
                    [CENTER_X, CENTER_Y + CENTER_CIRCLE_RADIUS],
                ], dtype=np.float32)

            return None

        elif class_name == "Half Field":
            if side_hint == "left":
                return np.array([
                    [0.0, 0.0],
                    [CENTER_X, 0.0],
                    [CENTER_X, PITCH_WIDTH],
                    [0.0, PITCH_WIDTH],
                ], dtype=np.float32)

            if side_hint == "right":
                return np.array([
                    [CENTER_X, 0.0],
                    [PITCH_LENGTH, 0.0],
                    [PITCH_LENGTH, PITCH_WIDTH],
                    [CENTER_X, PITCH_WIDTH],
                ], dtype=np.float32)

            return None

        elif class_name == "18Yard Circle":
            # The visible D-shape only spans the chord y-range, not the
            # full circle y-range. Compute the half-chord from the
            # circle geometry so the canvas bbox matches the image-side
            # rotated bbox produced by minAreaRect.
            chord_offset = abs(PENALTY_AREA_DEPTH - PENALTY_SPOT_DISTANCE)
            half_chord = np.sqrt(
                max(PENALTY_ARC_RADIUS ** 2 - chord_offset ** 2, 0.0)
            )

            y_top = CENTER_Y - half_chord
            y_bottom = CENTER_Y + half_chord

            if side_hint == "left":
                arc_outer_x = LEFT_PENALTY_SPOT_X + PENALTY_ARC_RADIUS

                return np.array([
                    [LEFT_PENALTY_X, y_top],
                    [arc_outer_x, y_top],
                    [arc_outer_x, y_bottom],
                    [LEFT_PENALTY_X, y_bottom],
                ], dtype=np.float32)

            if side_hint == "right":
                arc_outer_x = RIGHT_PENALTY_SPOT_X - PENALTY_ARC_RADIUS

                return np.array([
                    [arc_outer_x, y_top],
                    [RIGHT_PENALTY_X, y_top],
                    [RIGHT_PENALTY_X, y_bottom],
                    [arc_outer_x, y_bottom],
                ], dtype=np.float32)

            return None

        elif class_name == "18Yard":
            if side_hint == "left":
                return np.array([
                    [0.0, PENALTY_Y_TOP],
                    [LEFT_PENALTY_X, PENALTY_Y_TOP],
                    [LEFT_PENALTY_X, PENALTY_Y_BOTTOM],
                    [0.0, PENALTY_Y_BOTTOM],
                ], dtype=np.float32)

            if side_hint == "right":
                return np.array([
                    [RIGHT_PENALTY_X, PENALTY_Y_TOP],
                    [PITCH_LENGTH, PENALTY_Y_TOP],
                    [PITCH_LENGTH, PENALTY_Y_BOTTOM],
                    [RIGHT_PENALTY_X, PENALTY_Y_BOTTOM],
                ], dtype=np.float32)

            return None

        elif class_name == "5Yard":
            if side_hint == "left":
                return np.array([
                    [0.0, GOAL_AREA_Y_TOP],
                    [LEFT_GOAL_AREA_X, GOAL_AREA_Y_TOP],
                    [LEFT_GOAL_AREA_X, GOAL_AREA_Y_BOTTOM],
                    [0.0, GOAL_AREA_Y_BOTTOM],
                ], dtype=np.float32)

            if side_hint == "right":
                return np.array([
                    [RIGHT_GOAL_AREA_X, GOAL_AREA_Y_TOP],
                    [PITCH_LENGTH, GOAL_AREA_Y_TOP],
                    [PITCH_LENGTH, GOAL_AREA_Y_BOTTOM],
                    [RIGHT_GOAL_AREA_X, GOAL_AREA_Y_BOTTOM],
                ], dtype=np.float32)

            return None

        return None
    

    @staticmethod
    def suggest_side(image_quad, frame_w):
        cx = float(image_quad[:, 0].mean())
        return "left" if cx < frame_w / 2.0 else "right"
