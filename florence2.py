import os

from unittest.mock import patch
from typing import List, Dict, Any, Optional, Union, Tuple

os.environ['FIFTYONE_ALLOW_LEGACY_ORCHESTRATORS'] = 'true'

import numpy as np
import torch
from PIL import Image

import fiftyone as fo
from fiftyone import Model
from fiftyone.core.labels import Detection, Detections, Polyline, Polylines

from transformers.dynamic_module_utils import get_imports
from transformers import AutoModelForCausalLM, AutoProcessor

# Constants
DEFAULT_MODEL_PATH = "microsoft/Florence-2-base"

# Task definitions and parameter configurations
FLORENCE2_OPERATIONS = {
    "caption": {
        "params": {"detail_level": ["basic", "detailed", "more_detailed"]},
        "required": [],
        "task_mapping": {
            "detailed": "<DETAILED_CAPTION>",
            "more_detailed": "<MORE_DETAILED_CAPTION>",
            "basic": "<CAPTION>",
            None: "<CAPTION>"  # Default value
        }
    },
    "ocr": {
        "params": {"store_region_info": bool},
        "required": [],
        "task": "<OCR>",
        "region_task": "<OCR_WITH_REGION>"
    },
    "detection": {
        "params": {"detection_type": ["detection", "dense_region_caption", "region_proposal", "open_vocabulary_detection"],
                   "text_prompt": str},
        "required": [],
        "task_mapping": {
            "detection": "<OD>",
            "dense_region_caption": "<DENSE_REGION_CAPTION>",
            "region_proposal": "<REGION_PROPOSAL>",
            "open_vocabulary_detection": "<OPEN_VOCABULARY_DETECTION>",
            None: "<OD>"  # Default value
        }
    },
    "phrase_grounding": {
        "params": {"caption_field": str, "caption": str},
        "required": [],  # Will be validated in code
        "task": "<CAPTION_TO_PHRASE_GROUNDING>"
    },
    "segmentation": {
        "params": {"expression": str, "expression_field": str},
        "required": [],  # Will be validated in code
        "task": "<REFERRING_EXPRESSION_SEGMENTATION>"
    }
}

# Utility functions
def get_device():
    """Get the appropriate device for model inference."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def _convert_bbox(bbox, width, height):
    """Convert bounding box coordinates to FiftyOne format.
    
    Takes raw bounding box coordinates and converts them to normalized coordinates
    in FiftyOne's [x, y, width, height] format. Handles both standard rectangular
    bounding boxes (4 coordinates) and quadrilateral boxes (8 coordinates).

    Args:
        bbox: List of coordinates. Either [x1,y1,x2,y2] for rectangular boxes
              or [x1,y1,x2,y2,x3,y3,x4,y4] for quadrilateral boxes
        width: Width of the image in pixels
        height: Height of the image in pixels

    Returns:
        list: Normalized coordinates in format [x, y, width, height] where:
            - x,y is the top-left corner (normalized by image dimensions)
            - width,height are the box dimensions (normalized by image dimensions)
    """
    if len(bbox) == 4:
        # Standard rectangular box: convert from [x1,y1,x2,y2] to [x,y,w,h]
        # x1,y1 is top-left corner, x2,y2 is bottom-right corner
        return [
            bbox[0] / width,              # x coordinate (normalized)
            bbox[1] / height,             # y coordinate (normalized) 
            (bbox[2] - bbox[0]) / width,  # width (normalized)
            (bbox[3] - bbox[1]) / height  # height (normalized)
        ]
    else:
        # Quadrilateral box: find bounding rectangle that contains all points
        x1, y1, x2, y2, x3, y3, x4, y4 = bbox
        x_min = min(x1, x2, x3, x4)  # Leftmost x coordinate
        x_max = max(x1, x2, x3, x4)  # Rightmost x coordinate
        y_min = min(y1, y2, y3, y4)  # Topmost y coordinate
        y_max = max(y1, y2, y3, y4)  # Bottommost y coordinate

        return [
            x_min / width,               # x coordinate (normalized)
            y_min / height,              # y coordinate (normalized)
            (x_max - x_min) / width,     # width (normalized)
            (y_max - y_min) / height     # height (normalized)
        ]


class Florence2(Model):
    """A FiftyOne model for running the Florence-2 multimodal model on images.
    
    The Florence-2 model supports multiple vision-language tasks including:
    - Image captioning (with varying levels of detail)
    - OCR with region detection
    - Open vocabulary object detection
    - Phrase grounding (linking caption phrases to regions)
    - Referring expression segmentation
    
    Args:
        operation (str): Type of operation to perform. Must be one of: 
                        'caption', 'ocr', 'detection', 'phrase_grounding', 'segmentation'
        model_path (str, optional): Model path or HuggingFace repo name.
                                   Defaults to "microsoft/Florence-2-base".
        **kwargs: Operation-specific parameters:
            - caption: detail_level (str, optional) - "basic", "detailed", or "more_detailed"
            - ocr: store_region_info (bool, optional) - Whether to include region information
            - detection: detection_type (str, optional) - Type of detection to perform
                         text_prompt (str, optional) - Text prompt for open vocabulary detection
            - phrase_grounding: caption_field (str) or caption (str) - Caption source
            - segmentation: expression_field (str) or expression (str) - Referring expression
    
    Example::
        
        # Create a captioning model
        model = Florence2(operation="caption", detail_level="detailed")
        
        # Run detection
        model = Florence2(operation="detection")
        
        # Run phrase grounding on an existing caption field
        model = Florence2(operation="phrase_grounding", caption_field="my_captions")
    """

    def __init__(
        self, 
        operation: str,
        model_path: str = DEFAULT_MODEL_PATH,
        **kwargs
    ):
        """Initialize the Florence-2 model.
        
        Args:
            operation: Type of operation to perform
            model_path: Model path or HuggingFace repo name
            **kwargs: Operation-specific parameters
        
        Raises:
            ValueError: If the operation is invalid or required parameters are missing
        """
        self.operation = operation
        self.model_path = model_path
        
        # Validate operation
        if operation not in FLORENCE2_OPERATIONS:
            raise ValueError(f"Invalid operation: {operation}. Must be one of {list(FLORENCE2_OPERATIONS.keys())}")
        
        # Operation-specific validation
        if operation == "phrase_grounding":
            if "caption_field" not in kwargs and "caption" not in kwargs:
                raise ValueError("Either 'caption_field' or 'caption' must be provided for phrase_grounding operation")
        
        if operation == "segmentation":
            if "expression_field" not in kwargs and "expression" not in kwargs:
                raise ValueError("Either 'expression_field' or 'expression' must be provided for segmentation operation")
        
        self.params = kwargs

        # Set device
        self.device = get_device()

        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # Initialize model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            attn_implementation="sdpa", 
            trust_remote_code=True,
            device_map=self.device,
            torch_dtype=self.torch_dtype
            )

        self.processor = AutoProcessor.from_pretrained(
            model_path, 
            trust_remote_code=True
        )

    @property
    def media_type(self):
        """Get the media type supported by this model."""
        return "image"

    def _generate_and_parse(
        self,
        image: Image.Image,
        task: str,
        text_input: Optional[str] = None,
        max_new_tokens: int = 1024,
        num_beams: int = 3,
    ):
        """Generate and parse a response from the model.
        
        Args:
            image: The input image
            task: The task prompt to use
            text_input: Optional text input that includes the task
            max_new_tokens: Maximum new tokens to generate
            num_beams: Number of beams for beam search
            
        Returns:
            The parsed model output
        """
        text = task
        if text_input is not None:
            text = text_input
            
        inputs = self.processor(text=text, images=image, return_tensors="pt")
        
        # Move inputs to the device
        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].to(self.device, self.torch_dtype)

        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
        )
        generated_text = self.processor.batch_decode(
            generated_ids, 
            skip_special_tokens=False
        )[0]

        parsed_answer = self.processor.post_process_generation(
            generated_text, 
            task=task, 
            image_size=(image.width, image.height)
        )

        return parsed_answer

    def _extract_detections(self, parsed_answer, task, image):
        """Extracts object detections from the model's parsed output and converts them to FiftyOne format.
        
        Args:
            parsed_answer: Dict containing the parsed model output with bounding boxes and labels
            task: String specifying the task type - either "<OPEN_VOCABULARY_DETECTION>" or "<OCR_WITH_REGION>"
            image: PIL Image object used to get dimensions for normalizing coordinates
            
        Returns:
            A FiftyOne Detections object containing the extracted detections, where each detection has:
            - A label (either from model output or "object_N" if no label provided)
            - A normalized bounding box in [0,1] coordinates
        """
        # Choose the appropriate keys based on the task type
        label_key = (
            "bboxes_labels" if task == "<OPEN_VOCABULARY_DETECTION>" else "labels"
        )
        bbox_key = "quad_boxes" if task == "<OCR_WITH_REGION>" else "bboxes"
        
        # Extract bounding boxes and labels from the parsed output
        bboxes = parsed_answer[task][bbox_key]
        labels = parsed_answer[task][label_key]
        
        # Build list of FiftyOne Detection objects
        dets = []
        for i, (bbox, label) in enumerate(zip(bboxes, labels)):
            # Create Detection with either model label or fallback object_N label
            dets.append(
                Detection(
                    label=label if label else f"object_{i+1}",
                    bounding_box=_convert_bbox(bbox, image.width, image.height),
                )
            )
            
        # Return all detections wrapped in a FiftyOne Detections object
        return Detections(detections=dets)

    def _extract_polylines(self, parsed_answer, task, image):
        """Extract polylines from segmentation results and convert them to FiftyOne format.
        
        Takes the raw polygon coordinates from the model output and converts them into
        normalized coordinates relative to the image dimensions. Creates closed polylines
        that can be visualized as filled polygons in FiftyOne.
        
        Args:
            parsed_answer (dict): The parsed model output containing polygon coordinates
                in the format {task: {"polygons": [[[x1,y1,x2,y2,...]]]}
            task (str): The segmentation task that was performed
            image (PIL.Image): The input image used to normalize coordinates
            
        Returns:
            fiftyone.core.labels.Polylines: A FiftyOne Polylines object containing all
                the extracted polygons, where each polyline has:
                - points: List of (x,y) coordinates normalized to [0,1]
                - label: "object_N" where N is the polygon index
                - filled: True to render as filled polygon
                - closed: True to connect first/last points
            None: If no polygons were found in the parsed output
        """
        # Extract list of polygons from model output
        polygons = parsed_answer[task]["polygons"]
        if not polygons:
            return None

        polylines = []

        # Process each polygon
        for k, polygon in enumerate(polygons):
            # Process all contours for this polygon
            all_contours = []
            for contour in polygon:
                # Separate interleaved x,y coordinates and normalize by image dimensions
                x_points = [p for i, p in enumerate(contour) if i % 2 == 0]
                y_points = [p for i, p in enumerate(contour) if i % 2 != 0]
                x_points = [x / image.width for x in x_points]
                y_points = [y / image.height for y in y_points]

                # Convert to list of (x,y) tuples in a zigzag pattern
                xy_points = []
                curr_x = x_points[0]
                curr_y = y_points[0]
                xy_points.append((curr_x, curr_y))
                
                for i in range(1, len(x_points)):
                    curr_x = x_points[i]
                    xy_points.append((curr_x, curr_y))
                    curr_y = y_points[i] 
                    xy_points.append((curr_x, curr_y))

                # Close the contour
                xy_points.append((x_points[0], curr_y))
                all_contours.append(xy_points)

            # Create FiftyOne Polyline object with all contours
            polylines.append(
                Polyline(
                    points=all_contours,  # Now includes all contours for this polygon
                    label=f"object_{k+1}",
                    filled=True,
                    closed=True,
                )
            )
        # Return all polylines wrapped in a FiftyOne Polylines object
        return Polylines(polylines=polylines)

    def _predict_caption(self, image: Image.Image) -> str:
        """Generate a caption for an image.
        
        Args:
            image: The input image
            
        Returns:
            The generated caption
        """
        detail_level = self.params.get("detail_level", "basic")
        task_mapping = FLORENCE2_OPERATIONS["caption"]["task_mapping"]
        task = task_mapping.get(detail_level, task_mapping[None])  # Get appropriate task or default
            
        parsed_answer = self._generate_and_parse(image, task)
        return parsed_answer[task]

    def _predict_ocr(self, image: Image.Image) -> Union[str, Detections]:
        """Perform OCR on an image.
        
        Args:
            image: The input image
            
        Returns:
            OCR text string or Detections object with text regions (if store_region_info=True)
        """
        store_region_info = self.params.get("store_region_info", False)
        
        if store_region_info:
            task = FLORENCE2_OPERATIONS["ocr"]["region_task"]
            parsed_answer = self._generate_and_parse(image, task)
            return self._extract_detections(parsed_answer, task, image)
        else:
            task = FLORENCE2_OPERATIONS["ocr"]["task"]
            parsed_answer = self._generate_and_parse(image, task)
            return parsed_answer[task]

    def _predict_detection(self, image: Image.Image) -> Detections:
        """Detect objects in an image.
        
        Args:
            image: The input image
            
        Returns:
            Detections object with detected objects
        """
        detection_type = self.params.get("detection_type", None)
        text_prompt = self.params.get("text_prompt", None)
        
        task_mapping = FLORENCE2_OPERATIONS["detection"]["task_mapping"]
        task = task_mapping.get(detection_type, task_mapping[None])
        
        parsed_answer = self._generate_and_parse(image, task, text_input=text_prompt)
        return self._extract_detections(parsed_answer, task, image)

    def _predict_phrase_grounding(self, image: Image.Image) -> Detections:
        """Ground caption phrases in an image.
        
        Args:
            image: The input image
            
        Returns:
            Detections object with grounded phrases
        """
        task = FLORENCE2_OPERATIONS["phrase_grounding"]["task"]
        
        # Determine caption input
        if "caption" in self.params:
            caption = self.params["caption"]
        else:
            # caption_field will be resolved by the caller
            caption = self.params["caption_field"]
        
        text_input = f"{task}\n{caption}"
        
        parsed_answer = self._generate_and_parse(image, task, text_input=text_input)
        return self._extract_detections(parsed_answer, task, image)

    def _predict_segmentation(self, image: Image.Image) -> Optional[Polylines]:
        """Segment an object in an image based on a referring expression.
        
        Args:
            image: The input image
            
        Returns:
            Polylines segmentation mask for the referred object or None if no object found
        """
        task = FLORENCE2_OPERATIONS["segmentation"]["task"]
        
        # Determine expression input
        if "expression" in self.params:
            expression = self.params["expression"]
        else:
            # expression_field will be resolved by the caller
            expression = self.params["expression_field"] 
        
        text_input = f"{task}\nExpression: {expression}"
        
        parsed_answer = self._generate_and_parse(image, task, text_input=text_input)
        return self._extract_polylines(parsed_answer, task, image)

    def _predict(self, image: Image.Image) -> Any:
        """Process a single image with Florence-2.
        
        Args:
            image: The input image
            
        Returns:
            Operation result (type depends on operation)
        """
        prediction_methods = {
            "caption": self._predict_caption,
            "ocr": self._predict_ocr,
            "detection": self._predict_detection,
            "phrase_grounding": self._predict_phrase_grounding,
            "segmentation": self._predict_segmentation
        }
        
        predict_method = prediction_methods.get(self.operation)

        if predict_method is None:
            raise ValueError(f"Unknown operation: {self.operation}")
            
        return predict_method(image)

    def predict(self, image: np.ndarray) -> Any:
        """Process an image array with Florence-2.
        
        This method is called by FiftyOne's apply_model method.
        
        Args:
            image: numpy array image
            
        Returns:
            Operation result (type depends on operation)
        """
        pil_image = Image.fromarray(image)
        return self._predict(pil_image)

def run_florence2_model(
    dataset: fo.Dataset,
    operation: str,
    output_field: str,
    model_path: str = DEFAULT_MODEL_PATH,
    **kwargs
) -> None:
    """Apply Florence-2 operations to a FiftyOne dataset.
    
    Args:
        dataset: FiftyOne dataset to process
        operation: Type of operation to perform
        output_field: Field to store results in
        model_path: Model path or HuggingFace repo name
        **kwargs: Operation-specific parameters
    """
    # Handle field-based parameters
    if operation == "phrase_grounding" and "caption_field" in kwargs:
        # We'll need to process each sample to get the caption from the field
        model = Florence2(
            operation=operation,
            model_path=model_path,
            caption_field=kwargs["caption_field"]
        )
        
        for sample in dataset.iter_samples(autosave=True):
            caption = sample[kwargs["caption_field"]]
            # Override the caption parameter with the actual value
            model.params["caption"] = caption
            result = model.predict(np.array(Image.open(sample.filepath).convert("RGB")))
            sample[output_field] = result
            
    elif operation == "segmentation" and "expression_field" in kwargs:
        # We'll need to process each sample to get the expression from the field
        model = Florence2(
            operation=operation,
            model_path=model_path,
            expression_field=kwargs["expression_field"]
        )
        
        for sample in dataset.iter_samples(autosave=True):
            expression = sample[kwargs["expression_field"]]
            # Override the expression parameter with the actual value
            model.params["expression"] = expression
            result = model.predict(np.array(Image.open(sample.filepath).convert("RGB")))
            sample[output_field] = result
    else:
        # Standard apply_model workflow for parameters that don't depend on sample fields
        model = Florence2(
            operation=operation,
            model_path=model_path,
            **kwargs
        )
        dataset.apply_model(model, label_field=output_field)
