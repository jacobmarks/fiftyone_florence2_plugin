import os
from typing import List, Dict, Any, Optional, Union, Tuple

os.environ['FIFTYONE_ALLOW_LEGACY_ORCHESTRATORS'] = 'true'

import numpy as np
import torch
from PIL import Image

import fiftyone as fo
from fiftyone import Model
from fiftyone.core.labels import Detection, Detections, Polyline, Polylines

from transformers import AutoModelForCausalLM, AutoProcessor

# Constants
DEFAULT_MODEL_PATH = "microsoft/Florence-2-base-ft"

# Task definitions and parameter configurations
FLORENCE2_OPERATIONS = {
    "caption": {
        "params": {"detail_level": ["basic", "detailed", "more_detailed"]},
        "task_mapping": {
            "detailed": "<DETAILED_CAPTION>",
            "more_detailed": "<MORE_DETAILED_CAPTION>",
            "basic": "<CAPTION>",
            None: "<CAPTION>"  # Default value
        }
    },
    "ocr": {
        "params": {"store_region_info": bool},
        "task": "<OCR>",
        "region_task": "<OCR_WITH_REGION>"
    },
    "detection": {
        "params": {"detection_type": ["detection", "dense_region_caption", "region_proposal", "open_vocabulary_detection"],
                   "text_prompt": str},
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
        "task": "<CAPTION_TO_PHRASE_GROUNDING>"
    },
    "segmentation": {
        "params": {"expression": str, "expression_field": str},
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


def _convert_polyline(contour, width, height):
    """Convert polyline coordinates to FiftyOne format.
    
    Takes raw polyline coordinates and converts them to normalized coordinates
    in FiftyOne's format.

    Args:
        contour: List of interleaved x,y coordinates [x1,y1,x2,y2,...]
        width: Width of the image in pixels
        height: Height of the image in pixels

    Returns:
        list: List of (x,y) tuples representing normalized coordinates of the contour
    """
    # Initialize empty list to store the normalized coordinate pairs
    xy_points = []
    
    # Process the interleaved coordinates by taking them in pairs (x,y)
    # The interleaved format means all values at even indices are x-coordinates
    # and all values at odd indices are y-coordinates
    for i in range(0, len(contour), 2):
        # Check if we have both x and y coordinates available
        # This guards against malformed input where the list might have an odd length
        if i+1 < len(contour):
            # Extract the x-coordinate and normalize it to [0,1] range
            # Normalization makes coordinates independent of image dimensions
            x = contour[i] / width
            
            # Extract the corresponding y-coordinate and normalize it
            y = contour[i+1] / height
            
            # Add the normalized (x,y) pair to our result list
            # This maintains the original shape without adding artificial points
            xy_points.append((x, y))
    
    # Return the list of normalized coordinate pairs
    # This format is ready to be used with FiftyOne's Polyline class
    # The 'closed' parameter in FiftyOne will determine if the shape is closed, not this function
    return xy_points

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
                                   Defaults to "microsoft/Florence-2-base-ft".
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
        print(f"Using device: {self.device}")

        self.torch_dtype = torch.float16 if torch.cuda.is_available() else None

        # Initialize model
        if self.torch_dtype:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                trust_remote_code=True,
                device_map=self.device,
                torch_dtype=self.torch_dtype
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                trust_remote_code=True,
                device_map=self.device
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
        max_new_tokens: int = 4096,
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
        
        # Move inputs to device without dtype casting for MPS/CPU
        for key in inputs:
            if torch.is_tensor(inputs[key]):
                if torch.cuda.is_available() and key == "pixel_values":
                    inputs[key] = inputs[key].to(self.device, self.torch_dtype)
                else:
                    inputs[key] = inputs[key].to(self.device)

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
        
        Args:
            parsed_answer (dict): The parsed model output containing polygon coordinates
            task (str): The segmentation task that was performed
            image (PIL.Image): The input image used to normalize coordinates
            
        Returns:
            fiftyone.core.labels.Polylines: A FiftyOne Polylines object containing all
                the extracted polygons, or None if no polygons were found
        """
        polygons = parsed_answer[task]["polygons"]
        if not polygons:
            return None

        polylines = []

        # Process each polygon
        for k, polygon in enumerate(polygons):
            # Process all contours for this polygon
            all_contours = [
                _convert_polyline(contour, image.width, image.height)
                for contour in polygon
            ]

            # Create FiftyOne Polyline object with all contours
            polylines.append(
                Polyline(
                    points=all_contours,
                    label=f"object_{k+1}",
                    filled=True,
                    closed=True,
                )
            )

        return Polylines(polylines=polylines)

    def _predict_caption(self, image: Image.Image) -> str:
        """Generate a natural language caption describing the input image.
        
        This method uses the Florence-2 model to generate a descriptive caption for the image.
        The level of detail in the caption can be controlled via the "detail_level" parameter.

        Args:
            image: PIL Image object containing the image to be captioned
            
        Returns:
            str: A natural language caption describing the contents and context of the image
            
        Note:
            The detail_level parameter can be set to:
            - "basic": Short, simple caption (default)
            - "detailed": Longer, more descriptive caption
            - "dense": Very detailed caption with multiple aspects described
        """
        # Get the requested caption detail level, defaulting to "basic"
        detail_level = self.params.get("detail_level", "basic")
        
        # Get the mapping of detail levels to Florence-2 task specifications
        task_mapping = FLORENCE2_OPERATIONS["caption"]["task_mapping"]
        
        # Look up the appropriate task for the detail level, falling back to default if not found
        task = task_mapping.get(detail_level, task_mapping[None])
            
        # Generate the caption by running the model and parsing its output
        parsed_answer = self._generate_and_parse(image, task)
        
        # Extract and return just the caption text from the parsed response
        return parsed_answer[task]

    def _predict_ocr(self, image: Image.Image) -> Union[str, Detections]:
        """Perform Optical Character Recognition (OCR) on an input image.
        
        This method uses the Florence-2 model to detect and extract text from images.
        It can operate in two modes:
        1. Text extraction only - returns just the detected text
        2. Region-based OCR - returns text with bounding box coordinates
        
        Args:
            image (Image.Image): PIL Image object containing the image to perform OCR on
            
        Returns:
            Union[str, Detections]: Either:
                - A string containing all detected text (when store_region_info=False)
                - A Detections object containing text regions with bounding boxes
                  (when store_region_info=True)
                  
        Note:
            The behavior is controlled by the store_region_info parameter in self.params.
            Set store_region_info=True to get region/bounding box information.
        """
        # Check if region information should be included in output
        store_region_info = self.params.get("store_region_info", False)
        
        if store_region_info:
            # Use region-based OCR task that includes bounding box coordinates
            task = FLORENCE2_OPERATIONS["ocr"]["region_task"]
            parsed_answer = self._generate_and_parse(image, task)
            # Convert the parsed output into FiftyOne Detections format
            return self._extract_detections(parsed_answer, task, image)
        else:
            # Use basic OCR task that returns only text
            task = FLORENCE2_OPERATIONS["ocr"]["task"]
            parsed_answer = self._generate_and_parse(image, task)
            # Return just the extracted text string
            return parsed_answer[task]

    def _predict_detection(self, image: Image.Image) -> Detections:
        """Detect objects in an image using the Florence2 model.
        
        This method performs object detection on the input image. It supports two modes:
        1. Open vocabulary detection - Detects common objects without specific prompting
        2. Prompted detection - Detects objects matching a provided text prompt
        
        Args:
            image (Image.Image): PIL Image object containing the image to analyze
            
        Returns:
            Detections: FiftyOne Detections object containing the detected objects.
                       Each detection includes a label and bounding box coordinates.
                       
        Note:
            The detection behavior is controlled by two parameters in self.params:
            - detection_type: Controls the detection mode (open vocabulary vs prompted)
            - text_prompt: Optional text prompt specifying objects to detect
        """
        # Get detection parameters from self.params, defaulting to None if not specified
        detection_type = self.params.get("detection_type", None)
        text_prompt = self.params.get("text_prompt", None)
        
        # Look up the appropriate Florence2 task based on detection_type
        task_mapping = FLORENCE2_OPERATIONS["detection"]["task_mapping"]
        task = task_mapping.get(detection_type, task_mapping[None])  # Fall back to default if type not found
        
        # Run the model and parse its output, passing text_prompt if provided
        parsed_answer = self._generate_and_parse(image, task, text_input=text_prompt)
        
        # Convert the parsed model output into FiftyOne's Detections format
        return self._extract_detections(parsed_answer, task, image)

    def _predict_phrase_grounding(self, image: Image.Image) -> Detections:
        """Ground caption phrases in an image using the Florence2 model.
        
        This method performs phrase grounding by identifying regions in the image that
        correspond to specific phrases from a caption. It can use either a direct caption
        string or a caption stored in a sample field.

        Args:
            image (Image.Image): PIL Image object containing the image to analyze
            
        Returns:
            Detections: FiftyOne Detections object containing the grounded phrases.
                       Each detection includes the phrase text as the label and 
                       bounding box coordinates indicating the region in the image.

        Note:
            The caption source is controlled by parameters in self.params:
            - caption: Direct caption string to ground
            - caption_field: Name of sample field containing caption to ground
        """
        # Get the phrase grounding task configuration
        task = FLORENCE2_OPERATIONS["phrase_grounding"]["task"]
        
        # Determine caption input - either direct caption or field reference
        if "caption" in self.params:
            # Use directly provided caption string
            caption = self.params["caption"]
        else:
            # Use caption from specified field (resolved by caller)
            caption = self.params["caption_field"]
        
        # Format the input by combining task instruction and caption
        text_input = f"{task}\n{caption}"
        
        # Run model inference and parse the output
        parsed_answer = self._generate_and_parse(image, task, text_input=text_input)
        
        # Convert parsed output to FiftyOne Detections format
        return self._extract_detections(parsed_answer, task, image)

    def _predict_segmentation(self, image: Image.Image) -> Optional[Polylines]:
        """Segment an object in an image based on a referring expression.
        
        This method performs instance segmentation by generating a polygon mask around
        an object described by a natural language expression. The expression can be 
        provided directly or referenced from a sample field.

        Args:
            image (Image.Image): PIL Image object containing the image to analyze

        Returns:
            Optional[Polylines]: FiftyOne Polylines object containing the segmentation
                mask as a polygon, or None if no matching object is found in the image

        Note:
            The referring expression is controlled by parameters in self.params:
            - expression: Direct text string describing the object to segment
            - expression_field: Name of sample field containing the referring expression
        """
        # Get the segmentation task configuration from Florence2 operations
        task = FLORENCE2_OPERATIONS["segmentation"]["task"]
        
        # Determine the referring expression - either direct text or field reference
        if "expression" in self.params:
            # Use directly provided expression string
            expression = self.params["expression"]
        else:
            # Use expression from specified field (resolved by caller)
            expression = self.params["expression_field"] 
        
        # Format the input by combining task instruction and referring expression
        text_input = f"{task}\nExpression: {expression}"
        
        # Run model inference and parse the output
        parsed_answer = self._generate_and_parse(image, task, text_input=text_input)
        
        # Convert parsed output to FiftyOne Polylines format
        return self._extract_polylines(parsed_answer, task, image)

    def _predict(self, image: Image.Image) -> Any:
        """Process a single image with Florence2 model.
        
        This internal method handles routing the image to the appropriate prediction
        method based on the operation type (caption, OCR, detection, etc.) that was 
        specified when initializing the Florence2 class.

        Args:
            image (Image.Image): PIL Image object to process with the model
            
        Returns:
            Any: Operation-specific result type:
                - str for captioning
                - Detections for detection/phrase grounding 
                - Polylines for segmentation
                - List[Detection] for OCR
                
        Raises:
            ValueError: If self.operation is not one of the supported operation types
        """
        # Map operation names to their corresponding prediction methods
        prediction_methods = {
            "caption": self._predict_caption,          # Generate image caption
            "ocr": self._predict_ocr,                 # Optical character recognition
            "detection": self._predict_detection,      # Object detection
            "phrase_grounding": self._predict_phrase_grounding,  # Locate described objects
            "segmentation": self._predict_segmentation # Instance segmentation
        }
        
        # Get the prediction method for the requested operation
        predict_method = prediction_methods.get(self.operation)

        # Raise error if operation type is not supported
        if predict_method is None:
            raise ValueError(f"Unknown operation: {self.operation}")
            
        # Call the appropriate prediction method with the image
        return predict_method(image)

    def predict(self, image: np.ndarray) -> Any:
        """Process an image array with Florence2 model.
        
        This method serves as the main entry point when using FiftyOne's apply_model functionality.
        It converts the input numpy array to a PIL Image and routes it through the internal 
        prediction pipeline.
        
        Args:
            image (np.ndarray): Input image as a numpy array in RGB format with shape (H,W,3)
            
        Returns:
            Any: Operation-specific result type:
                - str for captioning operations
                - Detections for detection/phrase grounding operations
                - Polylines for segmentation operations 
                - List[Detection] for OCR operations
                
        Note:
            This method automatically handles conversion between numpy array and PIL Image formats.
            The specific return type depends on which operation was specified when initializing
            the Florence2 class.
        """
        # Convert numpy array to PIL Image format required by Florence2
        pil_image = Image.fromarray(image)
        
        # Route through internal prediction pipeline
        return self._predict(pil_image)

def run_florence2_model(
    dataset: fo.Dataset,
    operation: str,
    output_field: str,
    model_path: str = DEFAULT_MODEL_PATH,
    **kwargs
) -> None:
    """Apply Florence2 operations to a FiftyOne dataset.
    
    This function processes a FiftyOne dataset using the Florence2 model, supporting
    various vision tasks like captioning, detection, OCR, phrase grounding and segmentation.
    
    Args:
        dataset: FiftyOne dataset containing images to process
        operation: Type of operation to perform. Valid options are:
            - "caption": Generate image captions
            - "detection": Detect objects in images
            - "ocr": Perform optical character recognition
            - "phrase_grounding": Locate objects described by text
            - "segmentation": Perform instance segmentation
        output_field: Name of the field where results will be stored in the dataset
        model_path: HuggingFace model identifier or local path to model weights.
            Defaults to "microsoft/Florence-2-base-ft"
        **kwargs: Additional operation-specific parameters
    """
    # Handle phrase_grounding operation
    if operation == "phrase_grounding":
        if "caption" in kwargs:
            # Handle direct caption input
            model = Florence2(
                operation=operation,
                model_path=model_path,
                caption=kwargs["caption"]
            )
            # Apply model to entire dataset at once
            dataset.apply_model(model, label_field=output_field)
            
        elif "caption_field" in kwargs:
            # Handle per-sample captions from field
            model = Florence2(
                operation=operation,
                model_path=model_path,
                caption_field=kwargs["caption_field"]
            )
            
            for sample in dataset.iter_samples(autosave=True):
                caption = sample[kwargs["caption_field"]]
                model.params["caption"] = caption
                result = model.predict(np.array(Image.open(sample.filepath).convert("RGB")))
                sample[output_field] = result
                
        else:
            raise ValueError("Either 'caption' or 'caption_field' must be provided for phrase_grounding")
            
    # Handle segmentation operation    
    elif operation == "segmentation":
        if "expression" in kwargs:
            # Handle direct expression input
            model = Florence2(
                operation=operation,
                model_path=model_path,
                expression=kwargs["expression"]
            )
            # Apply model to entire dataset at once
            dataset.apply_model(model, label_field=output_field)
            
        elif "expression_field" in kwargs:
            # Handle per-sample expressions from field
            model = Florence2(
                operation=operation,
                model_path=model_path,
                expression_field=kwargs["expression_field"]
            )
            
            for sample in dataset.iter_samples(autosave=True):
                expression = sample[kwargs["expression_field"]]
                model.params["expression"] = expression
                result = model.predict(np.array(Image.open(sample.filepath).convert("RGB")))
                sample[output_field] = result
                
        else:
            raise ValueError("Either 'expression' or 'expression_field' must be provided for segmentation")
    
    # Handle detection operation with its specific parameters
    elif operation == "detection":
        # Create model with all detection parameters
        model = Florence2(
            operation=operation,
            model_path=model_path,
            **kwargs  # This passes detection_type and text_prompt if present
        )
        # Apply model to entire dataset at once
        dataset.apply_model(model, label_field=output_field)
        
    # Handle OCR operation with its specific parameters
    elif operation == "ocr":
        # Create model with all OCR parameters
        model = Florence2(
            operation=operation,
            model_path=model_path,
            **kwargs  # This passes store_region_info if present
        )
        # Apply model to entire dataset at once
        dataset.apply_model(model, label_field=output_field)
        
    # Handle caption and other operations
    else:
        # Generic case for other operations
        model = Florence2(
            operation=operation,
            model_path=model_path,
            **kwargs
        )
        dataset.apply_model(model, label_field=output_field)