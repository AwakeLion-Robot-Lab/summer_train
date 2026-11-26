import argparse
import json
import math
from pathlib import Path


def read_point(value):
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(type(coordinate) in (int, float) and math.isfinite(coordinate)
                   for coordinate in value)
    ):
        raise ValueError(f'invalid image point: {value!r}')
    return tuple(value)


def read_labels(path):
    labels = {}
    for row in json.loads(path.read_text()):
        frame = row['frame']
        if type(frame) is not int or frame < 0 or frame in labels:
            raise ValueError(f'invalid or duplicate labeled frame: {frame!r}')
        labels[frame] = [read_point(point) for point in row['centers']]
    if not labels:
        raise ValueError('provide at least one labeled frame')
    return labels


def count_matches(centers, detections, max_distance):
    candidates = [
        [index for distance, index in sorted(
            (math.dist(center, point), index)
            for index, point in enumerate(detections)
            if point is not None
        ) if distance <= max_distance]
        for center in centers
    ]
    assigned = {}

    def assign(center_index, seen):
        for detection_index in candidates[center_index]:
            if detection_index in seen:
                continue
            seen.add(detection_index)
            previous = assigned.get(detection_index)
            if previous is None or assign(previous, seen):
                assigned[detection_index] = center_index
                return True
        return False

    for center_index in range(len(centers)):
        assign(center_index, set())
    return len(assigned)


def score(audit, labels, max_distance, include_reobserved=False):
    rows = []
    with audit.open() as stream:
        for frame, line in enumerate(stream):
            record = json.loads(line)
            if record['frame'] != frame:
                raise ValueError('audit frames must be consecutive and start at zero')
            if frame not in labels:
                continue
            detections = []
            for feature in record['raw']:
                ellipse = feature.get('ellipse')
                center = ellipse['center'] if ellipse is not None else feature.get('center')
                detections.append(read_point(center) if center is not None else None)
            expected = len(labels[frame])
            if include_reobserved:
                detections.extend(read_point(ring['center'])
                                  for ring in record.get('reobserved_rings', []))
            matched = count_matches(labels[frame], detections, max_distance)
            rows.append(dict(frame=frame, expected=expected, matched=matched,
                             missed=expected - matched, false_positives=len(detections) - matched))

    if len(rows) != len(labels):
        raise ValueError('audit does not cover all labeled frames')
    expected = sum(row['expected'] for row in rows)
    matched = sum(row['matched'] for row in rows)
    dark = [row for row in rows if row['expected'] == 0]
    dark_with_detections = sum(row['false_positives'] > 0 for row in dark)
    detected = matched + sum(row['false_positives'] for row in rows)
    return dict(
        max_distance_px=max_distance,
        labeled_frames=len(rows),
        expected_blades=expected,
        matched_blades=matched,
        blade_recall=matched / expected if expected else None,
        blade_precision=matched / detected if detected else None,
        false_positive_detections=sum(row['false_positives'] for row in rows),
        dark_frames=len(dark),
        dark_frames_with_false_positives=dark_with_detections,
        dark_frame_false_positive_rate=dark_with_detections / len(dark) if dark else None,
        frames=rows,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--max-distance-px', type=float, required=True)
    parser.add_argument('--include-reobserved', action='store_true')
    args = parser.parse_args()
    if not math.isfinite(args.max_distance_px) or args.max_distance_px <= 0:
        parser.error('--max-distance-px must be finite and positive')
    try:
        report = score(args.audit, read_labels(args.labels), args.max_distance_px,
                       args.include_reobserved)
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
