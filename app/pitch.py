from constants import (
    CANVAS_H,
    CANVAS_W,
    DRAW_SCALE,
    BORDER,
    PITCH_LENGTH,
    PITCH_WIDTH,
    CENTER_X,
    CENTER_Y,
    CENTER_CIRCLE_RADIUS,
    PENALTY_AREA_DEPTH,
    PENALTY_AREA_WIDTH,
    PENALTY_Y_TOP,
    PENALTY_Y_BOTTOM,
    GOAL_AREA_DEPTH,
    GOAL_AREA_WIDTH,
    GOAL_AREA_Y_TOP,
    GOAL_AREA_Y_BOTTOM,
    PENALTY_SPOT_DISTANCE,
    PENALTY_ARC_RADIUS,
    LEFT_PENALTY_X,
    RIGHT_PENALTY_X,
    LEFT_GOAL_AREA_X,
    RIGHT_GOAL_AREA_X,
    LEFT_PENALTY_SPOT_X,
    RIGHT_PENALTY_SPOT_X,
    GREEN,
    WHITE,
    WHITE_LINE_THICKNESS,
    BALL_TRAIL_LENGTH,
    BALL_DOT_COLOR,
)
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



    def draw_players_on_pitch(
        self,
        img,
        player_points_pitch,
        labels=None,
        colors=None,
        default_color=(0, 0, 255),
    ):
        """
        Draw players on the pitch canvas.

        Args:
            img: Pitch canvas (H, W, 3).
            player_points_pitch: (N, 2) array of pitch coordinates in meters.
            labels: Optional list of labels for each player.
            colors: Optional list of BGR tuples, one per player.
                    If provided, each player dot uses its color.
            default_color: Fallback color when colors is None or missing.
        """
        out = img.copy()

        if player_points_pitch is None or len(player_points_pitch) == 0:
            return out

        pts = self.transform_coordinates_to_pixels(player_points_pitch)

        for i, (x, y) in enumerate(pts):
            x = int(x)
            y = int(y)

            if x < 0 or y < 0 or x >= out.shape[1] or y >= out.shape[0]:
                continue

            # Use per-player color if available
            if colors is not None and i < len(colors):
                color = colors[i]
            else:
                color = default_color

            cv2.circle(out, (x, y), 8, color, -1)
            # Outline for better visibility against pitch
            cv2.circle(out, (x, y), 10, (255, 255, 255), 1)

        return out

    def draw_team_legend(
        self,
        img,
        team1_color,
        team2_color,
        team1_label="Team 1",
        team2_label="Team 2",
    ):
        """
        Draw a color legend in the top-right corner of the pitch canvas.

        Args:
            img: Pitch canvas (H, W, 3).
            team1_color: BGR tuple for Team 1.
            team2_color: BGR tuple for Team 2.
            team1_label: Display name for Team 1.
            team2_label: Display name for Team 2.
        """
        out = img.copy()
        h, w = out.shape[:2]

        # Legend box dimensions
        box_x = w - 200
        box_y = 10
        box_w = 185
        box_h = 70

        # Semi-transparent background
        overlay = out.copy()
        cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, out, 0.5, 0, out)

        # Team 1 swatch
        cv2.circle(out, (box_x + 18, box_y + 22), 8, tuple(map(int, team1_color)), -1)
        cv2.circle(out, (box_x + 18, box_y + 22), 10, (255, 255, 255), 1)
        cv2.putText(out, team1_label, (box_x + 32, box_y + 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Team 2 swatch
        cv2.circle(out, (box_x + 18, box_y + 52), 8, tuple(map(int, team2_color)), -1)
        cv2.circle(out, (box_x + 18, box_y + 52), 10, (255, 255, 255), 1)
        cv2.putText(out, team2_label, (box_x + 32, box_y + 57),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        return out

    # ------------------------------------------------------------------
    # Ball visualization methods
    # ------------------------------------------------------------------
    def draw_ball_on_pitch(
        self,
        img: np.ndarray,
        ball_pitch_pt: np.ndarray,
        ball_color: tuple = BALL_DOT_COLOR,
    ) -> np.ndarray:
        """
        Draw the ball as a bright dot on the pitch canvas.

        Args:
            img: Pitch canvas (H, W, 3).
            ball_pitch_pt: (2,) array — ball position on pitch in meters.
            ball_color: BGR color for the ball dot.

        Returns:
            Canvas with ball dot drawn.
        """
        if ball_pitch_pt is None or len(ball_pitch_pt) < 2:
            return img

        out = img.copy()
        pt_px = self.transform_coordinates_to_pixels(
            ball_pitch_pt.reshape(1, 2)
        )[0]
        x, y = int(pt_px[0]), int(pt_px[1])

        if x < 0 or y < 0 or x >= out.shape[1] or y >= out.shape[0]:
            return out

        # Draw ball as a larger, brighter dot with white outline
        cv2.circle(out, (x, y), 10, ball_color, -1)
        cv2.circle(out, (x, y), 12, (255, 255, 255), 2)
        cv2.circle(out, (x, y), 14, (0, 0, 0), 1)

        return out

    def draw_ball_trajectory(
        self,
        img: np.ndarray,
        trajectory: list,
        max_trail: int = BALL_TRAIL_LENGTH,
    ) -> np.ndarray:
        """
        Draw a fading trajectory trail of ball positions on the pitch canvas.

        The trail fades from bright red (most recent) to orange (oldest),
        with line segments connecting consecutive positions.

        Args:
            img: Pitch canvas (H, W, 3).
            trajectory: List of (2,) pitch-coordinate arrays or [x, y] lists,
                        ordered from oldest to newest.
            max_trail: Maximum number of positions to display.

        Returns:
            Canvas with ball trajectory trail drawn.
        """
        if not trajectory:
            return img

        out = img.copy()

        # Take the most recent N positions
        recent = trajectory[-max_trail:]
        if len(recent) < 2:
            return out

        # Convert to pixel coordinates
        pts_np = np.array(recent, dtype=np.float32).reshape(-1, 2)
        pts_px = self.transform_coordinates_to_pixels(pts_np)

        # Draw trail as connected line segments with fading color
        n_segments = len(pts_px) - 1
        for i in range(n_segments):
            x1, y1 = int(pts_px[i][0]), int(pts_px[i][1])
            x2, y2 = int(pts_px[i + 1][0]), int(pts_px[i + 1][1])

            # Skip out-of-bounds
            if (x1 < 0 or y1 < 0 or x1 >= out.shape[1] or y1 >= out.shape[0] or
                x2 < 0 or y2 < 0 or x2 >= out.shape[1] or y2 >= out.shape[0]):
                continue

            # Fading color: older = dark orange, newer = bright red
            ratio = i / max(n_segments - 1, 1)
            r = int(0 + ratio * 255)        # 0 → 255
            g = int(165 * (1 - ratio))       # 165 → 0
            b = 0
            color = (b, g, r)

            # Line thickness: thicker for newer segments
            thickness = max(1, int(2 * ratio + 1))
            cv2.line(out, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

        # Draw small dots at each trajectory position
        trail_color = (0, 165, 255)  # Orange
        for i, (x, y) in enumerate(pts_px):
            xi, yi = int(x), int(y)
            if xi < 0 or yi < 0 or xi >= out.shape[1] or yi >= out.shape[0]:
                continue
            # Fade dot opacity: older = smaller
            dot_radius = max(1, int(2 * (i / max(len(pts_px) - 1, 1)) + 1))
            cv2.circle(out, (xi, yi), dot_radius, trail_color, -1)

        return out