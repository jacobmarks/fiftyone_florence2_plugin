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
from transformers import AutoModelForCausalLM, AutoProcessor

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
            donwloaded, as declared by the ``base_filename`` field of the
            manifest
        **kwargs: optional keyword arguments that configure how the model
            is loaded

    Returns:
        a :class:`fiftyone.core.models.Model`
    """

    # The directory containing this file
    model_dir = os.path.dirname(model_path)

    if torch.cuda.is_available():
        device="cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device="mps"
    else:
        device="cpu"
    
    print(f"Using device: {device}")

    torch_dtype = torch.float16 if torch.cuda.is_available() else None

    # Initialize model
    if torch_dtype:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            trust_remote_code=True,
            device_map=device,
            torch_dtype=torch_dtype
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            trust_remote_code=True,
            device_map=device
        )

    processor = AutoProcessor.from_pretrained(
        model_path, 
        trust_remote_code=True
    )

    return model, processor