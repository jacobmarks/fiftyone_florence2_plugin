## 🏛️ Florence2 Plugin

![florence2_plugin_demo](assets/florence2-plugin.gif)

### Plugin Overview

This plugin allows you to apply Florence2 directly to your FiftyOne datasets.

## Installation

If you haven't already, install FiftyOne:

```shell
pip install -U fiftyone transformers<=4.49 accelerate einops timm torch 
```

Then, install the plugin:

```shell
fiftyone plugins download https://github.com/jacobmarks/fiftyone_florence2_plugin
```

You can also install requirements via:

```shell
fiftyone plugins requirements @jacobmarks/florence2 --install
```

The Florence2 plugin integrates Microsoft's Florence2 Vision-Language Model with FiftyOne datasets, offering several powerful computer vision capabilities:

1. **Caption Generation** (`CaptionWithFlorence2`)
   - Generates image captions with different levels of detail
   - Customizable output field names
   - Supports both immediate and delegated execution

2. **Object Detection** (`DetectWithFlorence2`)
   - Multiple detection modes:
     - Standard object detection with default classes
     - Dense region captioning
     - Open vocabulary detection with custom text prompts
     - Region proposals
   - Flexible output field configuration

3. **OCR** (`OCRWithFlorence2`)
   - Performs text detection in images
   - Option to store region information (bounding boxes)
   - Configurable output fields

4. **Caption-to-Phrase Grounding** (`CaptionToPhraseGroundingWithFlorence2`)
   - Grounds specific phrases or captions by detecting relevant objects
   - Supports both direct text input and existing string fields from the dataset

5. **Referring Expression Segmentation** (`ReferringExpressionSegmentationWithFlorence2`)
   - Performs segmentation based on textual descriptions
   - Accepts either direct text input or references to existing dataset fields
   - Outputs segmentation masks for the described objects

All operators support:
- Custom model paths (defaults to "microsoft/Florence-2-base-ft")
- Delegated execution for resource-intensive tasks
- Flexible output field naming
- Integration with FiftyOne's dataset operations

## ℹ👨🏽‍💻 Refer to the [examples notebook](example_sdk_operators.ipynb) for detailed examples for how to run each operator via the FiftyOne SDK!


# Citation

```bibtext
@article{xiao2023florence,
  title={Florence-2: Advancing a unified representation for a variety of vision tasks},
  author={Xiao, Bin and Wu, Haiping and Xu, Weijian and Dai, Xiyang and Hu, Houdong and Lu, Yumao and Zeng, Michael and Liu, Ce and Yuan, Lu},
  journal={arXiv preprint arXiv:2311.06242},
  year={2023}
}
```