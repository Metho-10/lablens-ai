# LabLens AI — Workspace Object Audit

LabLens AI is a computer vision project built with Python,
Streamlit, and Ultralytics YOLOv8.

It analyzes uploaded images, detects supported objects,
displays bounding boxes, and generates object inventory reports.

## Features

- Image upload
- YOLOv8 object detection
- Adjustable confidence threshold
- Annotated detection images
- Object counts and confidence scores
- CSV report download
- Automated tests for report calculations

## Technologies

- Python
- Ultralytics YOLOv8
- Streamlit
- Pandas
- Pillow
- Pytest

## Setup

Install the dependencies:

    pip install -r requirements.txt

Run the application:

    python -m streamlit run app.py

Run the tests:

    python -m pytest -v

The pretrained model weights are downloaded when the model
is first loaded, so an internet connection may be needed.

## Limitations

The pretrained model only detects supported object categories.
It is not a specialized laboratory safety model.

## Purpose

This project demonstrates computer vision, web development,
automated testing, reporting, and Git-based version control.
