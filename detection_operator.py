import fiftyone as fo
import fiftyone.operators as foo
from fiftyone.operators import types

from .florence2 import  DEFAULT_MODEL_PATH

from .utils import _handle_calling, _BaseFlorence2Operator

# Specific operator classes
class DetectWithFlorence2(_BaseFlorence2Operator):
    """Operator for detecting objects with Florence-2."""
    
    def __init__(self):
        self.operation = "detection"
        self.operation_label = "detect objects in images with Florence-2"
        self.form_label = "Detect with Florence-2"
        self.form_description = "Detect objects in images using Florence-2"
    
    def _add_operation_inputs(self, ctx, inputs):
        # Detection type
        detection_task_choices = [
            "detection",
            "dense_region_caption",
            "region_proposal",
            "open_vocabulary_detection",
        ]

        radio_group = types.RadioGroup()
        for choice in detection_task_choices:
            radio_group.add_choice(choice, label=choice)

        inputs.enum(
            "detection_type",
            radio_group.values(),
            label="Detection type",
            description="The type of detection to perform",
            required=False,
            view=types.DropdownView(),
        )

        detection_task = ctx.params.get("detection_type", None)
        if detection_task is None:
            return

        if detection_task == "open_vocabulary_detection":
            inputs.str(
                "text_prompt",
                label="Text prompt",
                description="What do you want to detect?",
                required=True,
            )
            
        # Detection field
        inputs.str(
            "detection_field",
            label="Detection field",
            description="The field in which to store the detection results",
            required=False,
        )
    
    def _get_operation_kwargs(self, ctx):
        kwargs = {
            "detection_type": ctx.params.get("detection_type", None)
        }
        
        if ctx.params.get("detection_type") == "open_vocabulary_detection":
            kwargs["text_prompt"] = ctx.params.get("text_prompt", None)
            
        return kwargs
        
    def __call__(
        self, 
        sample_collection,
        output_field=None,
        detection_type="detection",  # Explicit parameter
        text_prompt=None,  # Explicit parameter
        model_path=DEFAULT_MODEL_PATH,
        delegate=False
    ):
        """Detect objects in images using Florence-2.
        
        Args:
            sample_collection: FiftyOne dataset or view to process
            output_field: Field to store detection results in
            detection_type: Type of detection to perform. Options: 
                           "detection", "dense_region_caption", 
                           "region_proposal", "open_vocabulary_detection"
            text_prompt: Text prompt for what to detect (required for "open_vocabulary_detection")
            model_path: Path to Florence-2 model
            delegate: Whether to delegate execution
            
        Returns:
            The operation result
        """
        kwargs = {"detection_type": detection_type}
        if text_prompt is not None:
            kwargs["text_prompt"] = text_prompt
            
        return _handle_calling(
            self.uri,
            sample_collection,
            model_path,
            self.operation,
            output_field,
            delegate,
            **kwargs
        )
