"""Small spatial association tracker; missing detections never reuse old coordinates."""

import math

from .models import box_iou


class DogTracker:
    def __init__(self, dog_index=0):
        self.dog_index = dog_index
        self.previous = None
        self.lost = 0

    def select(self, detections):
        if not detections:
            self.lost += 1
            return None, "no_dog"
        if self.previous is None:
            # Stable initial convention: index 0 is the largest dog, then decreasing area.
            ranked = sorted(detections, key=lambda d: (d[0][2] - d[0][0]) * (d[0][3] - d[0][1]), reverse=True)
            if self.dog_index >= len(ranked):
                self.lost += 1
                return None, "target_not_found"
            result = ranked[self.dog_index]
        else:
            x1, y1, x2, y2 = self.previous
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            diagonal = max(1, math.hypot(x2 - x1, y2 - y1))
            ranked = []
            for box, score in detections:
                distance = math.hypot((box[0] + box[2]) / 2 - cx, (box[1] + box[3]) / 2 - cy) / diagonal
                overlap = box_iou(box, self.previous)
                if overlap >= 0.03 or distance <= min(1.5, 0.75 + self.lost * 0.05):
                    ranked.append((overlap + 0.1 * score - 0.2 * distance, box, score))
            ranked.sort(key=lambda d: d[0], reverse=True)
            if not ranked:
                self.lost += 1
                return None, "target_lost"
            if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < 0.1:
                self.lost += 1
                return None, "ambiguous_dogs"
            _, box, score = ranked[0]
            result = box, score
        self.previous = result[0]
        self.lost = 0
        return result, "detected"
