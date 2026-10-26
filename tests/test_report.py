from reporting import summarize_detections, total_detections


def test_summarize_detections():
    detections = [
        {"Object": "bottle", "Confidence": 0.91},
        {"Object": "chair", "Confidence": 0.83},
        {"Object": "bottle", "Confidence": 0.76},
    ]

    assert summarize_detections(detections) == [
        {"Object": "bottle", "Count": 2},
        {"Object": "chair", "Count": 1},
    ]


def test_total_detections():
    detections = [
        {"Object": "bottle", "Confidence": 0.91},
        {"Object": "chair", "Confidence": 0.83},
    ]

    assert total_detections(detections) == 2


def test_empty_detections():
    assert summarize_detections([]) == []
    assert total_detections([]) == 0
