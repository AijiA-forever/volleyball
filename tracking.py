# -*- coding: utf-8 -*-
"""轻量多人跟踪：基于 IoU 的贪心匹配，为每个运动员分配稳定 ID。"""


class SimpleTracker:
    def __init__(self, iou_threshold=0.25, max_lost=30):
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self.tracks = {}
        self.next_id = 1

    @staticmethod
    def _iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _center(box):
        return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

    def update(self, boxes):
        """boxes: [(x1,y1,x2,y2), ...] 返回与 boxes 对齐的 ID 列表。"""
        boxes = [tuple(float(v) for v in b) for b in boxes]
        result = [None] * len(boxes)
        pairs = []
        for tid, track in self.tracks.items():
            for idx, box in enumerate(boxes):
                iou = self._iou(track["bbox"], box)
                if iou >= self.iou_threshold:
                    pairs.append((iou, tid, idx))
        pairs.sort(reverse=True)
        used_tracks, used_boxes = set(), set()
        for iou, tid, idx in pairs:
            if tid in used_tracks or idx in used_boxes:
                continue
            used_tracks.add(tid)
            used_boxes.add(idx)
            result[idx] = tid
            self.tracks[tid]["bbox"] = boxes[idx]
            self.tracks[tid]["lost"] = 0
        for idx, box in enumerate(boxes):
            if idx not in used_boxes:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {"bbox": box, "lost": 0}
                result[idx] = tid
        for tid in list(self.tracks.keys()):
            if tid not in used_tracks:
                self.tracks[tid]["lost"] += 1
                if self.tracks[tid]["lost"] > self.max_lost:
                    self.tracks.pop(tid, None)
        return result

    def reset(self):
        self.tracks = {}
        self.next_id = 1