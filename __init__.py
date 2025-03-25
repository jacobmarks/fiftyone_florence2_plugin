import os

os.environ['FIFTYONE_ALLOW_LEGACY_ORCHESTRATORS'] = 'true'

import fiftyone as fo
import fiftyone.operators as foo
from fiftyone.operators import types    

from .caption_operator import CaptionWithFlorence2
from .detection_operator import DetectWithFlorence2
from .ocr_operator import OCRWithFlorence2
from .grounding_operator import CaptionToPhraseGroundingWithFlorence2
from .segmentation_operator import ReferringExpressionSegmentationWithFlorence2

def register(plugin):
    """Register operators with the plugin."""
    # Register individual task operators
    plugin.register(CaptionWithFlorence2)
    plugin.register(OCRWithFlorence2)
    plugin.register(DetectWithFlorence2)
    plugin.register(CaptionToPhraseGroundingWithFlorence2)
    plugin.register(ReferringExpressionSegmentationWithFlorence2)
    

#### Remote model zoo functionality

import torch 
from huggingface_hub import snapshot_download
from .florence2 import Florence2

def download_model(model_name, model_path):
    """Downloads the model.

    Args:
        model_name: the name of the model to download, as declared by the
            ``base_name`` and optional ``version`` fields of the manifest
        model_path: the absolute filename or directory to which to download the
            model, as declared by the ``base_filename`` field of the manifest
    """
    snapshot_download(repo_id=model_name, local_dir=model_path)

def load_model(model_name, model_path, **kwargs):
    """Loads the model.

    Args:
        model_name: the name of the model to load, as declared by the
            ``base_name`` and optional ``version`` fields of the manifest
        model_path: the absolute filename or directory to which the model was
            downloaded, as declared by the ``base_filename`` field of the
            manifest
        **kwargs: must include 'operation' parameter and any other operation-specific
            parameters required by Florence2

    Returns:
        a :class:`fiftyone.core.models.Model`

    Raises:
        ValueError: if 'operation' is not provided in kwargs or is invalid
    """
    if 'operation' not in kwargs:
        raise ValueError("'operation' parameter is required for Florence2 model")

    valid_operations = ["caption", "ocr", "detection", "phrase_grounding", "segmentation"]
    operation = kwargs['operation']
    
    if operation not in valid_operations:
        raise ValueError(
            f"Invalid operation '{operation}'. Must be one of: {', '.join(valid_operations)}"
        )

    model = Florence2(
        model_path=model_path,
        **kwargs
    )

    return model