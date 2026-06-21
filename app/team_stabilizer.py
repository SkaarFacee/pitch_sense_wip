"""Sequence-level team stabilization.

The online ``TeamColorAnalyzer`` is intentionally stateful and optimized for
single-frame/video-preview behaviour. This module works on the whole processed
sequence: it links short ByteTrack fragments into canonical identities and then
assigns one stable team membership plus one explicit role per identity.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from constants import (
    TEAM0, TEAM1, NO_TEAM,
    ROLE_UNKNOWN, ROLE_OUTFIELD, ROLE_GK, ROLE_REF,
    PITCH_LENGTH, PITCH_WIDTH,
    LEFT_PENALTY_X, RIGHT_PENALTY_X,
    PENALTY_Y_TOP, PENALTY_Y_BOTTOM,
    LEFT_GOAL_AREA_X, RIGHT_GOAL_AREA_X,
    GOAL_AREA_Y_TOP, GOAL_AREA_Y_BOTTOM,
    GK_PENALTY_BOX_MIN_FRAC,
    GK_LEFTMOST_X_MARGIN, GK_RIGHTMOST_X_MARGIN,
    REF_SATURATION_THRESHOLD, TOUCHLINE_MARGIN_M,
    SHORTS_FEATURE_WEIGHT,
)


DEFAULT_TEAM_COLORS = [(255, 0, 0), (0, 0, 255)]
REF_COLOR = (0, 0, 0)
UNKNOWN_COLOR = (128, 128, 128)


def _as_int_dict(value: Any, value_parser: Callable[[Any], int]) -> Dict[int, int]:
    if not isinstance(value, dict):
        return {}
    out: Dict[int, int] = {}
    for k, v in value.items():
        try:
            out[int(k)] = int(value_parser(v))
        except Exception:
            continue
    return out


def _parse_team(value: Any) -> int:
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"team0", "team_0", "t1", "team 1", "display_team1", "0"}:
            return TEAM0
        if text in {"team1", "team_1", "t2", "team2", "team 2", "display_team2", "1"}:
            return TEAM1
        if text in {"none", "no_team", "noteam", "ref", "unknown", "-1"}:
            return NO_TEAM
    v = int(value)
    if v in (TEAM0, TEAM1):
        return v
    return NO_TEAM


def _parse_role(value: Any) -> int:
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"outfield", "outfielder", "player", "field", "0"}:
            return ROLE_OUTFIELD
        if text in {"gk", "goalkeeper", "keeper", "1"}:
            return ROLE_GK
        if text in {"ref", "referee", "official", "2"}:
            return ROLE_REF
        return ROLE_UNKNOWN
    v = int(value)
    if v in (ROLE_OUTFIELD, ROLE_GK, ROLE_REF):
        return v
    return ROLE_UNKNOWN


def _parse_bgr(value: Any) -> Optional[Tuple[int, int, int]]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("#") and len(text) == 7:
            try:
                r = int(text[1:3], 16)
                g = int(text[3:5], 16)
                b = int(text[5:7], 16)
                return (b, g, r)
            except Exception:
                return None
        try:
            value = json.loads(text)
        except Exception:
            parts = [p.strip() for p in text.replace(";", ",").split(",")]
            if len(parts) == 3:
                try:
                    return tuple(int(float(p)) for p in parts)  # type: ignore[return-value]
                except Exception:
                    return None
    try:
        vals = list(value)
        if len(vals) != 3:
            return None
        b, g, r = [int(max(0, min(255, round(float(v))))) for v in vals]
        return (b, g, r)
    except Exception:
        return None


@dataclass
class TeamCalibration:
    """Optional full-sequence calibration and manual overrides.

    Accepted dictionary shape is intentionally flexible so Streamlit can pass
    JSON directly:

    ``team0_bgr`` / ``team1_bgr`` or ``team_seed_bgr`` provide BGR seed colors.
    Override dictionaries map track/identity ids to team ids or role ids.
    """

    team_seed_bgr: Dict[int, Tuple[int, int, int]] = field(default_factory=dict)
    track_team_overrides: Dict[int, int] = field(default_factory=dict)
    identity_team_overrides: Dict[int, int] = field(default_factory=dict)
    track_role_overrides: Dict[int, int] = field(default_factory=dict)
    identity_role_overrides: Dict[int, int] = field(default_factory=dict)

    @classmethod
    def from_any(cls, value: Any) -> "TeamCalibration":
        if value is None or value == "":
            return cls()
        if isinstance(value, TeamCalibration):
            return value
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except Exception:
                return cls()
        if not isinstance(value, dict):
            return cls()

        seeds: Dict[int, Tuple[int, int, int]] = {}
        for key, team in (("team0_bgr", TEAM0), ("team_0_bgr", TEAM0),
                          ("team1_bgr", TEAM0), ("t1_bgr", TEAM0),
                          ("team2_bgr", TEAM1), ("team_1_bgr", TEAM1),
                          ("t2_bgr", TEAM1)):
            parsed = _parse_bgr(value.get(key))
            if parsed is not None:
                seeds[team] = parsed

        seed_map = value.get("team_seed_bgr") or value.get("team_seeds") or value.get("seed_bgr")
        if isinstance(seed_map, dict):
            for k, v in seed_map.items():
                try:
                    team = _parse_team(k)
                except Exception:
                    continue
                parsed = _parse_bgr(v)
                if team in (TEAM0, TEAM1) and parsed is not None:
                    seeds[team] = parsed

        return cls(
            team_seed_bgr=seeds,
            track_team_overrides=_as_int_dict(value.get("track_team_overrides"), _parse_team),
            identity_team_overrides=_as_int_dict(value.get("identity_team_overrides"), _parse_team),
            track_role_overrides=_as_int_dict(value.get("track_role_overrides"), _parse_role),
            identity_role_overrides=_as_int_dict(value.get("identity_role_overrides"), _parse_role),
        )

    def is_empty(self) -> bool:
        return not (self.team_seed_bgr or self.track_team_overrides
                    or self.identity_team_overrides or self.track_role_overrides
                    or self.identity_role_overrides)

    def team_override_for(self, identity_id: int, track_ids: Iterable[int]) -> Optional[int]:
        if identity_id in self.identity_team_overrides:
            return self.identity_team_overrides[identity_id]
        votes: Dict[int, int] = {}
        for tid in track_ids:
            if tid in self.track_team_overrides:
                team = self.track_team_overrides[tid]
                votes[team] = votes.get(team, 0) + 1
        if not votes:
            return None
        return max(votes.items(), key=lambda kv: kv[1])[0]

    def role_override_for(self, identity_id: int, track_ids: Iterable[int]) -> Optional[int]:
        if identity_id in self.identity_role_overrides:
            return self.identity_role_overrides[identity_id]
        votes: Dict[int, int] = {}
        for tid in track_ids:
            if tid in self.track_role_overrides:
                role = self.track_role_overrides[tid]
                votes[role] = votes.get(role, 0) + 1
        if not votes:
            return None
        return max(votes.items(), key=lambda kv: kv[1])[0]


@dataclass
class DetectionObservation:
    frame_idx: int
    detection_idx: int
    track_id: int
    bbox_xyxy: np.ndarray
    confidence: float
    feature: np.ndarray
    jersey_bgr: np.ndarray
    weight: float
    pitch_pt: Optional[np.ndarray] = None


@dataclass
class IdentityTrack:
    identity_id: int
    track_ids: List[int]
    observations: List[DetectionObservation]
    feature: Optional[np.ndarray] = None
    quality: float = 0.0
    gk_side: Optional[str] = None
    team_id: int = NO_TEAM
    role_id: int = ROLE_UNKNOWN
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StableAssignments:
    track_to_identity: Dict[int, int]
    identities: Dict[int, IdentityTrack]
    team_centroids_feat: np.ndarray
    team_centroids_bgr: List[Tuple[int, int, int]]
    diagnostics: Dict[str, Any]

    def identity_for_track(self, track_id: int) -> int:
        return int(self.track_to_identity.get(int(track_id), -1))

    def assignment_for_track(self, track_id: int) -> Tuple[int, int, int, float]:
        identity_id = self.identity_for_track(track_id)
        identity = self.identities.get(identity_id)
        if identity is None:
            return NO_TEAM, ROLE_UNKNOWN, -1, 0.0
        return int(identity.team_id), int(identity.role_id), int(identity.identity_id), float(identity.quality)

    def team_info_for_frame(self, track_ids: np.ndarray) -> dict:
        track_ids = np.asarray(track_ids, dtype=np.int32)
        team_ids = np.full(len(track_ids), NO_TEAM, dtype=np.int32)
        role_ids = np.full(len(track_ids), ROLE_UNKNOWN, dtype=np.int32)
        identity_ids = np.full(len(track_ids), -1, dtype=np.int32)
        track_quality = np.zeros(len(track_ids), dtype=np.float32)
        team_colors: List[Tuple[int, int, int]] = []

        t1 = self.team_centroids_bgr[0] if len(self.team_centroids_bgr) > 0 else DEFAULT_TEAM_COLORS[0]
        t2 = self.team_centroids_bgr[1] if len(self.team_centroids_bgr) > 1 else DEFAULT_TEAM_COLORS[1]
        for i, tid in enumerate(track_ids):
            team, role, identity_id, quality = self.assignment_for_track(int(tid))
            team_ids[i] = team
            role_ids[i] = role
            identity_ids[i] = identity_id
            track_quality[i] = quality
            if role == ROLE_REF:
                team_colors.append(REF_COLOR)
            elif team == TEAM0:
                team_colors.append(t1)
            elif team == TEAM1:
                team_colors.append(t2)
            else:
                team_colors.append(UNKNOWN_COLOR)

        soft_team_probs = np.zeros((len(track_ids), 2), dtype=np.float32)
        for i, team in enumerate(team_ids):
            if team in (TEAM0, TEAM1):
                soft_team_probs[i, int(team)] = 1.0

        return {
            "team_ids": team_ids,
            "role_ids": role_ids,
            "identity_ids": identity_ids,
            "team_colors": team_colors,
            "team1_bgr": t1,
            "team2_bgr": t2,
            "track_quality": track_quality,
            "soft_team_probs": soft_team_probs,
            "stabilizer_diagnostics": self.diagnostics,
        }


class _UnionFind:
    def __init__(self, values: Iterable[int]):
        self.parent = {int(v): int(v) for v in values}

    def find(self, x: int) -> int:
        x = int(x)
        parent = self.parent.setdefault(x, x)
        if parent != x:
            self.parent[x] = self.find(parent)
        return self.parent[x]

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if rb < ra:
            ra, rb = rb, ra
        self.parent[rb] = ra


class TeamSequenceStabilizer:
    """Full-sequence stabilizer for team membership and roles."""

    def __init__(self, calibration: Any = None,
                 feature_distance_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
                 max_link_gap: int = 14,
                 max_appearance_distance: float = 42.0,
                 max_pitch_distance_m: float = 14.0,
                 ambiguity_margin: float = 0.12):
        self.calibration = TeamCalibration.from_any(calibration)
        self.feature_distance_fn = feature_distance_fn or feature_distance
        self.max_link_gap = int(max_link_gap)
        self.max_appearance_distance = float(max_appearance_distance)
        self.max_pitch_distance_m = float(max_pitch_distance_m)
        self.ambiguity_margin = float(ambiguity_margin)
        self.observations: List[DetectionObservation] = []

    def add_frame_observations(self, frame_idx: int, track_ids: np.ndarray,
                               player_xyxy: np.ndarray, player_conf: Optional[np.ndarray],
                               features: Optional[np.ndarray], jersey_bgr: Optional[np.ndarray],
                               weights: Optional[np.ndarray],
                               player_pitch_pts: Optional[np.ndarray] = None) -> None:
        n = len(player_xyxy) if player_xyxy is not None else 0
        if n == 0:
            return
        tids = np.asarray(track_ids, dtype=np.int32) if track_ids is not None and len(track_ids) == n else np.arange(n, dtype=np.int32)
        conf = np.asarray(player_conf, dtype=np.float32) if player_conf is not None and len(player_conf) == n else np.ones(n, dtype=np.float32)
        feats = np.asarray(features, dtype=np.float32) if features is not None and len(features) == n else np.zeros((n, 6), dtype=np.float32)
        bgrs = np.asarray(jersey_bgr, dtype=np.float32) if jersey_bgr is not None and len(jersey_bgr) == n else feats[:, :3]
        ws = np.asarray(weights, dtype=np.float32) if weights is not None and len(weights) == n else conf
        pitch = np.asarray(player_pitch_pts, dtype=np.float32) if player_pitch_pts is not None and len(player_pitch_pts) == n else None

        for i in range(n):
            pp = pitch[i].copy() if pitch is not None else None
            self.observations.append(DetectionObservation(
                frame_idx=int(frame_idx),
                detection_idx=int(i),
                track_id=int(tids[i]),
                bbox_xyxy=np.asarray(player_xyxy[i], dtype=np.float32).copy(),
                confidence=float(conf[i]),
                feature=np.asarray(feats[i], dtype=np.float32).reshape(-1)[:6].copy(),
                jersey_bgr=np.asarray(bgrs[i], dtype=np.float32).reshape(-1)[:3].copy(),
                weight=float(max(0.0, min(1.0, ws[i]))),
                pitch_pt=pp,
            ))

    def fit(self) -> StableAssignments:
        if not self.observations:
            return StableAssignments(
                track_to_identity={}, identities={},
                team_centroids_feat=np.zeros((2, 6), dtype=np.float32),
                team_centroids_bgr=list(DEFAULT_TEAM_COLORS),
                diagnostics={
                    "identity_count": 0,
                    "track_fragment_count": 0,
                    "linked_fragments": [],
                    "rejected_links": [],
                    "validation": {"ok": True, "switches": []},
                },
            )

        fragments = self._build_fragments()
        track_to_identity, groups, link_diag = self._link_fragments(fragments)
        identities = self._build_identities(groups, fragments)
        centroids_feat, centroids_bgr, cluster_diag = self._resolve_identity_assignments(identities)

        diagnostics = {
            "identity_count": len(identities),
            "track_fragment_count": len(fragments),
            "linked_fragments": link_diag["accepted"],
            "rejected_links": link_diag["rejected"],
            "team_counts": self._count_by(identities.values(), "team_id"),
            "role_counts": self._count_by(identities.values(), "role_id"),
            "no_team_identities": [int(i.identity_id) for i in identities.values() if i.team_id == NO_TEAM],
            "referee_identities": [int(i.identity_id) for i in identities.values() if i.role_id == ROLE_REF],
            "goalkeeper_identities": [int(i.identity_id) for i in identities.values() if i.role_id == ROLE_GK],
            "calibration": {
                "enabled": not self.calibration.is_empty(),
                "team_seed_bgr": {int(k): tuple(map(int, v)) for k, v in self.calibration.team_seed_bgr.items()},
                "track_team_overrides": dict(self.calibration.track_team_overrides),
                "identity_team_overrides": dict(self.calibration.identity_team_overrides),
                "track_role_overrides": dict(self.calibration.track_role_overrides),
                "identity_role_overrides": dict(self.calibration.identity_role_overrides),
            },
            "cluster": cluster_diag,
        }
        diagnostics["validation"] = validate_identity_assignments(identities.values())

        return StableAssignments(
            track_to_identity=track_to_identity,
            identities=identities,
            team_centroids_feat=centroids_feat,
            team_centroids_bgr=centroids_bgr,
            diagnostics=diagnostics,
        )

    @staticmethod
    def _count_by(identities: Iterable[IdentityTrack], attr: str) -> Dict[int, int]:
        out: Dict[int, int] = {}
        for identity in identities:
            key = int(getattr(identity, attr))
            out[key] = out.get(key, 0) + 1
        return out

    def _build_fragments(self) -> Dict[int, List[DetectionObservation]]:
        fragments: Dict[int, List[DetectionObservation]] = {}
        for obs in self.observations:
            fragments.setdefault(int(obs.track_id), []).append(obs)
        for obs_list in fragments.values():
            obs_list.sort(key=lambda o: (o.frame_idx, o.detection_idx))
        return fragments

    def _link_fragments(self, fragments: Dict[int, List[DetectionObservation]]
                        ) -> Tuple[Dict[int, int], Dict[int, List[int]], Dict[str, list]]:
        tids = sorted(fragments.keys())
        uf = _UnionFind(tids)
        intervals = {tid: (fragments[tid][0].frame_idx, fragments[tid][-1].frame_idx) for tid in tids}
        candidates = self._candidate_links(fragments)

        by_src: Dict[int, List[dict]] = {}
        by_dst: Dict[int, List[dict]] = {}
        for cand in candidates:
            by_src.setdefault(cand["src"], []).append(cand)
            by_dst.setdefault(cand["dst"], []).append(cand)
        for bucket in list(by_src.values()) + list(by_dst.values()):
            bucket.sort(key=lambda c: c["score"])

        accepted: List[dict] = []
        rejected: List[dict] = []

        def _ambiguous(bucket: List[dict], best: dict) -> bool:
            if len(bucket) < 2:
                return False
            second = bucket[1]
            return (float(second["score"]) - float(best["score"])) < self.ambiguity_margin

        def _groups_overlap(root_a: int, root_b: int) -> bool:
            a_tids = [t for t in tids if uf.find(t) == root_a]
            b_tids = [t for t in tids if uf.find(t) == root_b]
            for a in a_tids:
                a0, a1 = intervals[a]
                for b in b_tids:
                    b0, b1 = intervals[b]
                    if not (a1 < b0 or b1 < a0):
                        return True
            return False

        for cand in sorted(candidates, key=lambda c: c["score"]):
            src_best = by_src.get(cand["src"], [None])[0]
            dst_best = by_dst.get(cand["dst"], [None])[0]
            if src_best is not cand or dst_best is not cand:
                item = dict(cand)
                item["reason"] = "not_mutual_best"
                rejected.append(item)
                continue
            if _ambiguous(by_src[cand["src"]], cand) or _ambiguous(by_dst[cand["dst"]], cand):
                item = dict(cand)
                item["reason"] = "ambiguous"
                rejected.append(item)
                continue
            ra = uf.find(cand["src"])
            rb = uf.find(cand["dst"])
            if ra == rb:
                continue
            if _groups_overlap(ra, rb):
                item = dict(cand)
                item["reason"] = "group_time_overlap"
                rejected.append(item)
                continue
            uf.union(ra, rb)
            accepted.append(dict(cand))

        raw_groups: Dict[int, List[int]] = {}
        for tid in tids:
            raw_groups.setdefault(uf.find(tid), []).append(tid)
        ordered = sorted(raw_groups.values(), key=lambda group: (min(intervals[t][0] for t in group), min(group)))
        groups: Dict[int, List[int]] = {identity_id: sorted(group) for identity_id, group in enumerate(ordered)}
        track_to_identity = {tid: identity_id for identity_id, group in groups.items() for tid in group}
        return track_to_identity, groups, {"accepted": accepted, "rejected": rejected}

    def _candidate_links(self, fragments: Dict[int, List[DetectionObservation]]) -> List[dict]:
        summaries = {tid: self._fragment_summary(obs) for tid, obs in fragments.items()}
        candidates: List[dict] = []
        for src, a in summaries.items():
            for dst, b in summaries.items():
                if src == dst:
                    continue
                gap = int(b["start"] - a["end"])
                if gap <= 0 or gap > self.max_link_gap:
                    continue

                appearance = self.feature_distance_fn(a["feature"], b["feature"])
                if appearance > self.max_appearance_distance:
                    continue

                center_dist = _center_distance_px(a, b, gap)
                max_center = max(70.0, 0.95 * max(a["height"], b["height"]) + 4.0 * gap)
                if center_dist > max_center:
                    continue

                pitch_dist = None
                if a["last_pitch"] is not None and b["first_pitch"] is not None:
                    pitch_dist = float(np.linalg.norm(a["last_pitch"] - b["first_pitch"]))
                    if pitch_dist > self.max_pitch_distance_m + 0.35 * gap:
                        continue

                score = (
                    appearance / max(self.max_appearance_distance, 1e-6)
                    + center_dist / max(max_center, 1e-6)
                    + 0.03 * gap
                )
                if pitch_dist is not None:
                    score += 0.35 * pitch_dist / max(self.max_pitch_distance_m, 1e-6)
                candidates.append({
                    "src": int(src), "dst": int(dst), "gap": gap,
                    "appearance": round(float(appearance), 3),
                    "center_distance_px": round(float(center_dist), 3),
                    "pitch_distance_m": round(float(pitch_dist), 3) if pitch_dist is not None else None,
                    "score": round(float(score), 4),
                })
        return candidates

    def _fragment_summary(self, observations: List[DetectionObservation]) -> dict:
        first = observations[0]
        last = observations[-1]
        feature, quality = _identity_feature(observations)
        centers = [_bbox_center(o.bbox_xyxy) for o in observations[-3:]]
        frames = [o.frame_idx for o in observations[-3:]]
        velocity = np.zeros(2, dtype=np.float32)
        if len(centers) >= 2 and frames[-1] > frames[0]:
            velocity = (centers[-1] - centers[0]) / float(frames[-1] - frames[0])
        return {
            "start": int(first.frame_idx),
            "end": int(last.frame_idx),
            "first_center": _bbox_center(first.bbox_xyxy),
            "last_center": _bbox_center(last.bbox_xyxy),
            "velocity": velocity,
            "first_pitch": first.pitch_pt,
            "last_pitch": last.pitch_pt,
            "height": float(max(1.0, np.median([o.bbox_xyxy[3] - o.bbox_xyxy[1] for o in observations]))),
            "feature": feature,
            "quality": quality,
        }

    def _build_identities(self, groups: Dict[int, List[int]],
                          fragments: Dict[int, List[DetectionObservation]]) -> Dict[int, IdentityTrack]:
        identities: Dict[int, IdentityTrack] = {}
        for identity_id, track_ids in groups.items():
            obs: List[DetectionObservation] = []
            for tid in track_ids:
                obs.extend(fragments[tid])
            obs.sort(key=lambda o: (o.frame_idx, o.detection_idx))
            feature, quality = _identity_feature(obs)
            gk_side = _gk_side_for_observations(obs)
            identities[identity_id] = IdentityTrack(
                identity_id=int(identity_id),
                track_ids=[int(t) for t in track_ids],
                observations=obs,
                feature=feature,
                quality=quality,
                gk_side=gk_side,
                diagnostics={
                    "frames_seen": len(obs),
                    "frame_start": int(obs[0].frame_idx) if obs else None,
                    "frame_end": int(obs[-1].frame_idx) if obs else None,
                    "gk_side": gk_side,
                },
            )
        return identities

    def _resolve_identity_assignments(self, identities: Dict[int, IdentityTrack]
                                      ) -> Tuple[np.ndarray, List[Tuple[int, int, int]], dict]:
        if not identities:
            return np.zeros((2, 6), dtype=np.float32), list(DEFAULT_TEAM_COLORS), {"source": "empty"}

        role_overrides = {
            identity_id: self.calibration.role_override_for(identity_id, identity.track_ids)
            for identity_id, identity in identities.items()
        }
        prelim_gk_ids = {
            identity_id for identity_id, identity in identities.items()
            if role_overrides.get(identity_id) == ROLE_GK
            or (role_overrides.get(identity_id) is None and identity.gk_side is not None)
        }
        ref_override_ids = {
            identity_id for identity_id, role in role_overrides.items()
            if role == ROLE_REF
        }

        centroid_pool = [
            identity for identity_id, identity in identities.items()
            if identity.feature is not None
            and identity_id not in ref_override_ids
            and identity_id not in prelim_gk_ids
        ]
        if len(centroid_pool) < 2:
            centroid_pool = [
                identity for identity_id, identity in identities.items()
                if identity.feature is not None and identity_id not in ref_override_ids
            ]

        centroids_feat, centroids_bgr, cluster_diag = self._team_centroids(centroid_pool)

        # Stable team means from non-GK/non-ref identities. GK membership can
        # then use long-term defensive-side evidence when kit colour is unique.
        provisional_team: Dict[int, int] = {}
        for identity_id, identity in identities.items():
            if identity.feature is None:
                continue
            provisional_team[identity_id] = int(np.argmin([
                self.feature_distance_fn(identity.feature, c) for c in centroids_feat
            ]))

        team_mean_x = _team_mean_x(identities, provisional_team, exclude_ids=prelim_gk_ids | ref_override_ids)

        for identity_id, identity in identities.items():
            role_override = role_overrides.get(identity_id)
            team_override = self.calibration.team_override_for(identity_id, identity.track_ids)

            if role_override is not None:
                role = role_override
            elif identity.gk_side is not None:
                role = ROLE_GK
            else:
                role = ROLE_OUTFIELD if identity.feature is not None else ROLE_UNKNOWN

            if role == ROLE_REF:
                team = NO_TEAM
            elif team_override is not None:
                team = team_override
            elif identity.feature is None:
                team = NO_TEAM
            elif role == ROLE_GK:
                team = _team_for_goalkeeper(identity, centroids_feat, team_mean_x, self.feature_distance_fn)
            else:
                nearest = int(np.argmin([self.feature_distance_fn(identity.feature, c) for c in centroids_feat]))
                if _looks_like_ref(identity, centroids_feat, self.feature_distance_fn):
                    team = NO_TEAM
                    role = ROLE_REF
                else:
                    team = nearest

            if team not in (TEAM0, TEAM1):
                team = NO_TEAM
                if role not in (ROLE_REF, ROLE_UNKNOWN):
                    role = ROLE_UNKNOWN
            elif role == ROLE_UNKNOWN:
                role = ROLE_OUTFIELD

            identity.team_id = int(team)
            identity.role_id = int(role)
            identity.diagnostics.update({
                "team_override": team_override,
                "role_override": role_override,
                "assigned_team": int(team),
                "assigned_role": int(role),
            })

        return centroids_feat, centroids_bgr, cluster_diag

    def _team_centroids(self, identity_pool: List[IdentityTrack]
                        ) -> Tuple[np.ndarray, List[Tuple[int, int, int]], dict]:
        seed0 = self.calibration.team_seed_bgr.get(TEAM0)
        seed1 = self.calibration.team_seed_bgr.get(TEAM1)
        if seed0 is not None and seed1 is not None:
            feat = np.zeros((2, 6), dtype=np.float32)
            feat[0, :3] = np.array(seed0, dtype=np.float32)
            feat[1, :3] = np.array(seed1, dtype=np.float32)
            return feat, [seed0, seed1], {"source": "calibration"}

        features = [identity.feature for identity in identity_pool if identity.feature is not None]
        if len(features) == 0:
            feat = np.zeros((2, 6), dtype=np.float32)
            feat[0, :3] = np.array(DEFAULT_TEAM_COLORS[0], dtype=np.float32)
            feat[1, :3] = np.array(DEFAULT_TEAM_COLORS[1], dtype=np.float32)
            return feat, list(DEFAULT_TEAM_COLORS), {"source": "default"}
        if len(features) == 1:
            feat = np.zeros((2, 6), dtype=np.float32)
            feat[0] = np.asarray(features[0], dtype=np.float32)
            feat[1, :3] = _complement_bgr(feat[0, :3])
            return _sort_centroids(feat), _centroids_to_bgr(_sort_centroids(feat)), {"source": "single_identity"}

        arr = np.asarray(features, dtype=np.float32)
        c0, c1, diag = self._select_pair_centroids(arr)
        centroids = np.stack([c0, c1]).astype(np.float32)
        for _ in range(8):
            labels = _nearest_labels(arr, centroids, self.feature_distance_fn)
            new_centroids = centroids.copy()
            for team in (TEAM0, TEAM1):
                group = arr[labels == team]
                if len(group) > 0:
                    new_centroids[team] = np.median(group, axis=0)
            if np.allclose(new_centroids, centroids, atol=1e-3):
                break
            centroids = new_centroids

        centroids = _sort_centroids(centroids)
        for team, seed in ((TEAM0, seed0), (TEAM1, seed1)):
            if seed is not None:
                centroids[team, :3] = np.array(seed, dtype=np.float32)
        return centroids, _centroids_to_bgr(centroids), diag

    def _select_pair_centroids(self, arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        best = None
        max_team_dist = 62.0
        n = len(arr)
        for i in range(n):
            for j in range(i + 1, n):
                pair = np.stack([arr[i], arr[j]]).astype(np.float32)
                pair_dist = self.feature_distance_fn(pair[0], pair[1])
                if pair_dist < 12.0:
                    continue
                labels = _nearest_labels(arr, pair, self.feature_distance_fn)
                dists = np.array([
                    self.feature_distance_fn(arr[k], pair[int(labels[k])])
                    for k in range(n)
                ], dtype=np.float32)
                covered = dists <= max_team_dist
                counts = [int(np.sum((labels == team) & covered)) for team in (TEAM0, TEAM1)]
                score = (int(np.sum(covered)), min(counts), -float(np.mean(dists[covered])) if np.any(covered) else -999.0)
                if best is None or score > best[0]:
                    best = (score, pair[0].copy(), pair[1].copy(), counts, float(pair_dist))
        if best is None:
            # Fallback: deterministic farthest pair.
            max_d = -1.0
            bi, bj = 0, 1
            for i in range(n):
                for j in range(i + 1, n):
                    d = self.feature_distance_fn(arr[i], arr[j])
                    if d > max_d:
                        max_d = d
                        bi, bj = i, j
            return arr[bi].copy(), arr[bj].copy(), {"source": "farthest_pair", "pair_distance": round(float(max_d), 3)}
        _, c0, c1, counts, pair_dist = best
        return c0, c1, {"source": "coverage_pair", "covered_counts": counts,
                        "pair_distance": round(float(pair_dist), 3)}


def feature_distance(f1: np.ndarray, f2: np.ndarray) -> float:
    f1 = np.asarray(f1, dtype=np.float32).reshape(-1)
    f2 = np.asarray(f2, dtype=np.float32).reshape(-1)
    if len(f1) < 6:
        f1 = np.pad(f1, (0, 6 - len(f1)))
    if len(f2) < 6:
        f2 = np.pad(f2, (0, 6 - len(f2)))
    d_j = float(np.linalg.norm(f1[:3] - f2[:3]))
    if max(float(np.max(f1[3:6])), float(np.max(f2[3:6]))) > 5.0:
        d_s = float(np.linalg.norm(f1[3:6] - f2[3:6]))
        return float(np.sqrt(d_j * d_j + (SHORTS_FEATURE_WEIGHT * d_s) ** 2))
    return d_j


def validate_identity_assignments(identities: Iterable[IdentityTrack]) -> dict:
    switches = []
    for identity in identities:
        emitted = {int(identity.team_id)}
        if len(emitted) > 1:
            switches.append({"identity_id": int(identity.identity_id), "team_ids": sorted(emitted)})
    return {"ok": len(switches) == 0, "switches": switches}


def validate_frame_team_infos(frame_team_infos: Iterable[dict]) -> dict:
    seen: Dict[int, set] = {}
    for frame_info in frame_team_infos:
        ids = np.asarray(frame_info.get("identity_ids", []), dtype=np.int32)
        teams = np.asarray(frame_info.get("team_ids", []), dtype=np.int32)
        for identity_id, team in zip(ids, teams):
            if int(identity_id) < 0:
                continue
            seen.setdefault(int(identity_id), set()).add(int(team))
    switches = [
        {"identity_id": identity_id, "team_ids": sorted(team_ids)}
        for identity_id, team_ids in sorted(seen.items())
        if len(team_ids) > 1
    ]
    return {"ok": len(switches) == 0, "switches": switches}


def _identity_feature(observations: List[DetectionObservation]) -> Tuple[Optional[np.ndarray], float]:
    if not observations:
        return None, 0.0
    valid = [o for o in observations if o.feature is not None and len(o.feature) >= 3 and o.weight >= 0.04]
    if not valid:
        valid = observations
    arr = np.array([np.pad(o.feature.reshape(-1), (0, max(0, 6 - len(o.feature))))[:6] for o in valid], dtype=np.float32)
    weights = np.array([max(0.01, float(o.weight)) for o in valid], dtype=np.float32)
    feat = np.array([_weighted_median(arr[:, dim], weights) for dim in range(6)], dtype=np.float32)
    quality = float(max(0.0, min(1.0, np.mean(weights))))
    return feat, quality


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    if len(values) == 0:
        return 0.0
    order = np.argsort(values)
    values = values[order]
    weights = np.maximum(weights[order], 0.0)
    total = float(np.sum(weights))
    if total <= 1e-6:
        return float(np.median(values))
    cutoff = total * 0.5
    idx = int(np.searchsorted(np.cumsum(weights), cutoff, side="left"))
    idx = min(max(idx, 0), len(values) - 1)
    return float(values[idx])


def _bbox_center(bbox: np.ndarray) -> np.ndarray:
    return np.array([(float(bbox[0]) + float(bbox[2])) * 0.5,
                     (float(bbox[1]) + float(bbox[3])) * 0.5], dtype=np.float32)


def _center_distance_px(a: dict, b: dict, gap: int) -> float:
    predicted = a["last_center"] + a["velocity"] * float(max(0, gap))
    return float(np.linalg.norm(predicted - b["first_center"]))


def _gk_side_for_observations(observations: List[DetectionObservation]) -> Optional[str]:
    pts = [o.pitch_pt for o in observations if o.pitch_pt is not None]
    if len(pts) < 3:
        return None
    left_pen = right_pen = left_goal = right_goal = 0
    on_pitch_x: List[float] = []
    for pt in pts:
        x = float(pt[0])
        y = float(pt[1])
        if -3 <= x <= PITCH_LENGTH + 3 and -3 <= y <= PITCH_WIDTH + 3:
            on_pitch_x.append(x)
        if x <= LEFT_PENALTY_X and PENALTY_Y_TOP <= y <= PENALTY_Y_BOTTOM:
            left_pen += 1
        if x >= RIGHT_PENALTY_X and PENALTY_Y_TOP <= y <= PENALTY_Y_BOTTOM:
            right_pen += 1
        if x <= LEFT_GOAL_AREA_X and GOAL_AREA_Y_TOP <= y <= GOAL_AREA_Y_BOTTOM:
            left_goal += 1
        if x >= RIGHT_GOAL_AREA_X and GOAL_AREA_Y_TOP <= y <= GOAL_AREA_Y_BOTTOM:
            right_goal += 1
    total = max(len(pts), 1)
    left_score = (left_pen + 0.75 * left_goal) / total
    right_score = (right_pen + 0.75 * right_goal) / total
    min_frac = max(0.35, float(GK_PENALTY_BOX_MIN_FRAC) * 0.75)
    if left_score >= min_frac and left_score >= right_score:
        return "left"
    if right_score >= min_frac and right_score > left_score:
        return "right"
    if len(on_pitch_x) >= 6:
        med_x = float(np.median(on_pitch_x))
        if med_x <= LEFT_PENALTY_X + GK_LEFTMOST_X_MARGIN:
            return "left"
        if med_x >= RIGHT_PENALTY_X - GK_RIGHTMOST_X_MARGIN:
            return "right"
    return None


def _team_for_goalkeeper(identity: IdentityTrack, centroids: np.ndarray,
                         team_mean_x: Dict[int, float],
                         distance_fn: Callable[[np.ndarray, np.ndarray], float]) -> int:
    if identity.gk_side == "left":
        if TEAM0 in team_mean_x and TEAM1 in team_mean_x:
            return TEAM0 if team_mean_x[TEAM0] <= team_mean_x[TEAM1] else TEAM1
    elif identity.gk_side == "right":
        if TEAM0 in team_mean_x and TEAM1 in team_mean_x:
            return TEAM0 if team_mean_x[TEAM0] >= team_mean_x[TEAM1] else TEAM1
    if identity.feature is None:
        return NO_TEAM
    return int(np.argmin([distance_fn(identity.feature, c) for c in centroids]))


def _team_mean_x(identities: Dict[int, IdentityTrack], provisional_team: Dict[int, int],
                 exclude_ids: set) -> Dict[int, float]:
    xs_by_team: Dict[int, List[float]] = {TEAM0: [], TEAM1: []}
    for identity_id, identity in identities.items():
        if identity_id in exclude_ids:
            continue
        team = provisional_team.get(identity_id)
        if team not in (TEAM0, TEAM1):
            continue
        for obs in identity.observations:
            if obs.pitch_pt is None:
                continue
            x = float(obs.pitch_pt[0])
            y = float(obs.pitch_pt[1])
            if -3 <= x <= PITCH_LENGTH + 3 and -3 <= y <= PITCH_WIDTH + 3:
                xs_by_team[team].append(x)
    return {team: float(np.mean(xs)) for team, xs in xs_by_team.items() if xs}


def _looks_like_ref(identity: IdentityTrack, centroids: np.ndarray,
                    distance_fn: Callable[[np.ndarray, np.ndarray], float]) -> bool:
    if identity.feature is None or len(centroids) < 2:
        return False
    dists = [distance_fn(identity.feature, c) for c in centroids]
    min_dist = float(min(dists))
    jersey = identity.feature[:3]
    colorfulness = float(np.max(jersey) - np.min(jersey))
    sat = _bgr_saturation(jersey)
    touchline_frac = _touchline_fraction(identity.observations)
    if min_dist > 78.0:
        return True
    if touchline_frac >= 0.55 and min_dist > 38.0:
        return True
    if sat < REF_SATURATION_THRESHOLD and colorfulness < 35.0 and min_dist > 45.0:
        return True
    return False


def _touchline_fraction(observations: List[DetectionObservation]) -> float:
    pts = [o.pitch_pt for o in observations if o.pitch_pt is not None]
    if not pts:
        return 0.0
    n = 0
    for pt in pts:
        y = float(pt[1])
        if y <= TOUCHLINE_MARGIN_M or y >= PITCH_WIDTH - TOUCHLINE_MARGIN_M:
            n += 1
    return n / max(len(pts), 1)


def _bgr_saturation(bgr: np.ndarray) -> float:
    sample = np.array([[[float(bgr[0]), float(bgr[1]), float(bgr[2])]]], dtype=np.uint8)
    try:
        return float(cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)[0, 0, 1])
    except Exception:
        return float(np.max(bgr) - np.min(bgr))


def _nearest_labels(arr: np.ndarray, centroids: np.ndarray,
                    distance_fn: Callable[[np.ndarray, np.ndarray], float]) -> np.ndarray:
    labels = []
    for row in arr:
        labels.append(int(np.argmin([distance_fn(row, c) for c in centroids])))
    return np.asarray(labels, dtype=np.int32)


def _sort_centroids(centroids: np.ndarray) -> np.ndarray:
    if len(centroids) < 2:
        return centroids
    keys = [_bgr_sort_key(c[:3]) for c in centroids]
    order = np.argsort(keys)
    return centroids[order].astype(np.float32)


def _bgr_sort_key(bgr: np.ndarray) -> float:
    sample = np.array([[[float(bgr[0]), float(bgr[1]), float(bgr[2])]]], dtype=np.uint8)
    try:
        hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)[0, 0]
        return float(hsv[0]) * 1000.0 + float(hsv[2])
    except Exception:
        return float(bgr[2]) * 1_000_000.0 + float(bgr[1]) * 1_000.0 + float(bgr[0])


def _centroids_to_bgr(centroids: np.ndarray) -> List[Tuple[int, int, int]]:
    out: List[Tuple[int, int, int]] = []
    for c in centroids[:2]:
        vals = np.clip(c[:3], 0, 255)
        out.append((int(vals[0]), int(vals[1]), int(vals[2])))
    while len(out) < 2:
        out.append(DEFAULT_TEAM_COLORS[len(out)])
    return out


def _complement_bgr(bgr: np.ndarray) -> np.ndarray:
    sample = np.array([[[float(bgr[0]), float(bgr[1]), float(bgr[2])]]], dtype=np.uint8)
    try:
        hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[0, 0, 0] = (hsv[0, 0, 0] + 90.0) % 180.0
        hsv[0, 0, 1] = max(hsv[0, 0, 1], 160.0)
        hsv[0, 0, 2] = max(hsv[0, 0, 2], 140.0)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)[0, 0].astype(np.float32)
    except Exception:
        return np.clip(255.0 - bgr, 0, 255).astype(np.float32)
