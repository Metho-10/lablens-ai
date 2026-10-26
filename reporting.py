from collections import Counter


def summarize_detections(detection_rows):
    """Count how many times each object appears."""

    counts = Counter(
        row["Object"] for row in detection_rows
    )

    return [
        {"Object": name, "Count": count}
        for name, count in sorted(counts.items())
    ]


def total_detections(detection_rows):
    """Return the total number of detected objects."""

    return len(detection_rows)
