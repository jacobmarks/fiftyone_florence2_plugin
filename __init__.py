import os
from typing import List, Dict, Any, Optional, Union, Tuple

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
    

# For compatibility with previous versions
def florence2_activator():
    """Check if required dependencies are installed."""
    from importlib.util import find_spec
    return (find_spec("transformers") is not None and 
            find_spec("einops") is not None and 
            find_spec("timm") is not None and 
            find_spec("torch") is not None and
            find_spec("PIL") is not None)