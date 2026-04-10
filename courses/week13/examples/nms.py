import numpy as np

def iou(box1, box2):
    # box format: [x1, y1, x2, y2]
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)

    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area_box1 + area_box2 - intersection

    return intersection / union

def nms(boxes, scores, iou_threshold=0.5):
    boxes = np.array(boxes)
    scores = np.array(scores)

    # Urutkan index berdasarkan skor dari besar ke kecil
    indices = scores.argsort()[::-1]

    keep = []

    while len(indices) > 0:
        current = indices[0]
        keep.append(current)

        remaining = indices[1:]
        filtered_indices = []

        for idx in remaining:
            overlap = iou(boxes[current], boxes[idx])
            if overlap <= iou_threshold:
                filtered_indices.append(idx)

        indices = np.array(filtered_indices)

    return keep

# Contoh data
boxes = [
    [10, 10, 50, 50],   # Kotak A
    [12, 12, 48, 48],   # Kotak B
    [14, 14, 52, 52],   # Kotak C
    [60, 60, 100, 100]  # Kotak D
]

scores = [0.95, 0.90, 0.82, 0.40]

selected = nms(boxes, scores, iou_threshold=0.5)

print("Kotak yang dipertahankan:")
for i in selected:
    print(f"Kotak {chr(65+i)}: {boxes[i]}, skor={scores[i]}")