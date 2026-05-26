from constants import * 
import numpy as np 
import cv2
class PitchArtist():

    def transform_coordinates_to_pixels(self,points_pitch):
        pts = points_pitch.copy()
        pts= pts* DRAW_SCALE + BORDER
        return pts.astype(np.int32)


    def draw_pitch_base(self):
        img = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
        img[:] = GREEN
        outer = self.transform_coordinates_to_pixels(np.array([
            [0.0, 0.0],
            [PITCH_LENGTH, 0.0],
            [PITCH_LENGTH, PITCH_WIDTH],
            [0.0, PITCH_WIDTH],
        ], dtype=np.float32))
        cv2.polylines(img, [outer], True, WHITE, WHITE_LINE_THICKNESS) # Drawing the pitch border 

        # Drawing the halfline
        p1 = self.transform_coordinates_to_pixels(np.array([CENTER_X, 0.0]))
        p2 = self.transform_coordinates_to_pixels(np.array([CENTER_X, PITCH_WIDTH]))
        cv2.line(img, tuple(p1), tuple(p2), WHITE, WHITE_LINE_THICKNESS) 

        center = self.transform_coordinates_to_pixels(np.array([CENTER_X, CENTER_Y]))
        cv2.circle(img, tuple(center), int(CENTER_CIRCLE_RADIUS * DRAW_SCALE), WHITE, WHITE_LINE_THICKNESS)
        cv2.circle(img, tuple(center), DRAW_SCALE//2, WHITE, -1)

        left_d_box = self.transform_coordinates_to_pixels(np.array([
            [0.0, PENALTY_Y_TOP],
            [LEFT_PENALTY_X, PENALTY_Y_TOP],
            [LEFT_PENALTY_X, PENALTY_Y_BOTTOM],
            [0.0, PENALTY_Y_BOTTOM],
        ]))

        right_d_box = self.transform_coordinates_to_pixels(np.array([
            [RIGHT_PENALTY_X, PENALTY_Y_TOP],
            [PITCH_LENGTH, PENALTY_Y_TOP],
            [PITCH_LENGTH, PENALTY_Y_BOTTOM],
            [RIGHT_PENALTY_X, PENALTY_Y_BOTTOM],
        ]))

        [cv2.polylines(img, [x], True, WHITE, WHITE_LINE_THICKNESS) for x in [left_d_box,right_d_box]]

        left_goal_box = self.transform_coordinates_to_pixels(np.array([
            [0.0, GOAL_AREA_Y_TOP],
            [LEFT_GOAL_AREA_X, GOAL_AREA_Y_TOP],
            [LEFT_GOAL_AREA_X, GOAL_AREA_Y_BOTTOM],
            [0.0, GOAL_AREA_Y_BOTTOM],
        ]))

        right_goal_box = self.transform_coordinates_to_pixels(np.array([
            [RIGHT_GOAL_AREA_X, GOAL_AREA_Y_TOP],
            [PITCH_LENGTH, GOAL_AREA_Y_TOP],
            [PITCH_LENGTH, GOAL_AREA_Y_BOTTOM],
            [RIGHT_GOAL_AREA_X, GOAL_AREA_Y_BOTTOM],
        ]))

        [cv2.polylines(img, [x], True, WHITE, WHITE_LINE_THICKNESS) for x in [left_goal_box,right_goal_box]]

        for point in [[LEFT_PENALTY_SPOT_X, CENTER_Y], [RIGHT_PENALTY_SPOT_X, CENTER_Y]]:
            pt = self.transform_coordinates_to_pixels(np.array([point], dtype=np.float32))[0]
            cv2.circle(img, tuple(pt), DRAW_SCALE//2, WHITE, -1)
        # D-box arcs
        arc_radius_px = int(PENALTY_ARC_RADIUS * DRAW_SCALE)
        left_spot = self.transform_coordinates_to_pixels(np.array([LEFT_PENALTY_SPOT_X, CENTER_Y]))
        right_spot = self.transform_coordinates_to_pixels(np.array([RIGHT_PENALTY_SPOT_X, CENTER_Y]))
        theta = np.degrees(np.arccos((LEFT_PENALTY_X - LEFT_PENALTY_SPOT_X) / PENALTY_ARC_RADIUS))
        cv2.ellipse(img,tuple(left_spot),(arc_radius_px, arc_radius_px),0,-theta,theta,WHITE,WHITE_LINE_THICKNESS) 
        cv2.ellipse(img,tuple(right_spot),(arc_radius_px, arc_radius_px),0,180-theta,180+theta,WHITE,WHITE_LINE_THICKNESS) 
        return img



    def draw_players_on_pitch(self,img, player_points_pitch, labels=None, color=(0, 0, 255)):
        out = img.copy()

        if player_points_pitch is None or len(player_points_pitch) == 0:
            return out

        pts = self.transform_coordinates_to_pixels(player_points_pitch)

        for i, (x, y) in enumerate(pts):
            x = int(x)
            y = int(y)

            if x < 0 or y < 0 or x >= out.shape[1] or y >= out.shape[0]:
                continue

            cv2.circle(out, (x, y), 8, color, -1)

            label = str(i) if labels is None else str(labels[i])
            cv2.putText(
                out,
                label,
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return out