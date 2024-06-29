## Florence-2 Plugin

### Plugin Overview

This plugin allows you to apply Florence-2 directly to your FiftyOne datasets.

## Installation

If you haven't already, install FiftyOne:

```shell
pip install -U fiftyone transformers accelerate
```

Then, install the plugin:

```shell
fiftyone plugins download https://github.com/jacobmarks/fiftyone_florence2_plugin
```

## Operators

### `caption_with_florence2`

Generate captions in three levels of detail

### `detect_with_florence2`

- Detect objects using Florence-2's default classes
- Dense region captioning
- Open vocabulary object detection with text input
- Region proposals

### `ocr_with_florence2`

- Detect text in images, with or without bounding boxes

### `caption_to_phrase_grounding_with_florence2`

- Ground an input phrase (caption) by detecting the relevant objects in the image. You can either specify the caption directly or use any `fo.StringField` on the dataset as the caption field.

### `referring_expression_segmentation_with_florence2`

- Segment the image based on the input referring expression. You can either specify the expression directly or use any `fo.StringField` on the dataset as the expression field.

Happy exploring!
