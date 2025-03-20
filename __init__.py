import os
from typing import List, Dict, Any, Optional, Union, Tuple

os.environ['FIFTYONE_ALLOW_LEGACY_ORCHESTRATORS'] = 'true'


import fiftyone as fo
import fiftyone.operators as foo

from fiftyone.operators import types    

from .utils import run_florence2_model,  _model_choice_inputs, _execution_mode, _handle_calling

from .caption_operator import CaptionWithFlorence2
from .detection_operator import DetectWithFlorence2
from .ocr_operator import OCRWithFlorence2
from .grounding_operator import CaptionToPhraseGroundingWithFlorence2
from .segmentation_operator import ReferringExpressionSegmentationWithFlorence2


DEFAULT_MODEL_PATH = "microsoft/Florence-2-base"

# Combined operator for all Florence-2 tasks
class Florence2Operator(foo.Operator):
    """Combined operator for all Florence-2 tasks."""
    
    @property
    def config(self):
        return foo.OperatorConfig(
            name="florence2",
            label="Run Florence-2",
            description="Run the Florence-2 model on your Dataset!",
            dynamic=True,
            icon="/assets/florence2-icon.svg",  # Replace with actual icon
        )
    
    def resolve_delegation(self, ctx):
        return ctx.params.get("delegate", False)
    
    def resolve_input(self, ctx):
        inputs = types.Object()
        
        # Operation selection
        operation_dropdown = types.Dropdown(label="What would you like to use Florence-2 for?")
        
        operations = {
            "caption": "Generate image captions",
            "ocr": "Perform OCR on images",
            "detection": "Detect objects in images",
            "phrase_grounding": "Ground phrases in captions",
            "segmentation": "Segment objects via referring expressions"
        }
        
        for k, v in operations.items():
            operation_dropdown.add_choice(k, label=v)
            
        inputs.enum(
            "operation",
            values=operation_dropdown.values(),
            label="Florence-2 Tasks",
            description="Select from one of the supported tasks.",
            view=operation_dropdown,
            required=True
        )
        
        # Model selection
        _model_choice_inputs(ctx, inputs)
        
        # Operation-specific inputs
        chosen_operation = ctx.params.get("operation")
        
        if chosen_operation == "caption":
            detail_level_choices = ["basic", "detailed", "more_detailed"]
            radio_group = types.RadioGroup()

            for choice in detail_level_choices:
                radio_group.add_choice(choice, label=choice)

            inputs.enum(
                "detail_level",
                radio_group.values(),
                label="Detail level",
                description="The level of detail to include in the caption",
                required=False,
                view=types.DropdownView(),
            )
            
        elif chosen_operation == "ocr":
            inputs.bool(
                "store_region_info",
                default=False,
                label="Store region information?",
                description="Check this box to represent OCR detections with bounding boxes",
                view=types.CheckboxView(),
            )
            
        elif chosen_operation == "detection":
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
            if detection_task == "open_vocabulary_detection":
                inputs.str(
                    "text_prompt",
                    label="Text prompt",
                    description="What do you want to detect?",
                    required=True,
                )
                
        elif chosen_operation == "phrase_grounding":
            input_choices = ["caption_field", "caption"]
            radio_group = types.RadioGroup()
            for choice in input_choices:
                radio_group.add_choice(choice, label=choice)

            inputs.enum(
                "caption_input",
                radio_group.values(),
                label="Caption input",
                description="The input to use for grounding phrases",
                required=True,
                default="caption",
                view=types.RadioView(),
            )

            input_type = ctx.params.get("caption_input", None)
            if input_type == "caption_field":
                candidate_fields = list(
                    ctx.dataset.get_field_schema(ftype=fo.StringField).keys()
                )
                if "filepath" in candidate_fields:
                    candidate_fields.remove("filepath")

                field_radio_group = types.RadioGroup()

                for field in candidate_fields:
                    field_radio_group.add_choice(field, label=field)

                inputs.enum(
                    "caption_field",
                    field_radio_group.values(),
                    label="Caption field",
                    description="The field to use as the caption",
                    required=True,
                    view=types.DropdownView(),
                )

            else:
                inputs.str(
                    "caption",
                    label="Caption",
                    description="The caption to use for grounding phrases",
                    required=True,
                )
                
        elif chosen_operation == "segmentation":
            input_choices = ["expression_field", "expression"]
            radio_group = types.RadioGroup()
            for choice in input_choices:
                radio_group.add_choice(choice, label=choice)

            inputs.enum(
                "expression_input",
                radio_group.values(),
                label="Referring expression input",
                description="The referring expression to use for segmentation",
                required=True,
                default="expression",
                view=types.RadioView(),
            )

            input_type = ctx.params.get("expression_input", None)
            if input_type == "expression_field":
                candidate_fields = list(
                    ctx.dataset.get_field_schema(ftype=fo.StringField).keys()
                )
                if "filepath" in candidate_fields:
                    candidate_fields.remove("filepath")

                field_radio_group = types.RadioGroup()

                for field in candidate_fields:
                    field_radio_group.add_choice(field, label=field)

                inputs.enum(
                    "expression_field",
                    field_radio_group.values(),
                    label="Expression field",
                    description="The field to use as the referring expression",
                    required=True,
                    view=types.DropdownView(),
                )

            else:
                inputs.str(
                    "expression",
                    label="Referring expression",
                    description="The referring expression to use for segmentation",
                    required=True,
                )
        
        # Output field
        inputs.str(
            "output_field",            
            required=True,
            label="Output Field",
            description="Name of the field to store the results in."
        )
        
        # Execution mode
        _execution_mode(ctx, inputs)
        
        inputs.view_target(ctx)
        
        return types.Property(inputs)
    
    def execute(self, ctx):
        dataset = ctx.dataset
        view = ctx.target_view()
        model_path = ctx.params.get("model_path", DEFAULT_MODEL_PATH)
        operation = ctx.params.get("operation")
        output_field = ctx.params.get("output_field")
        
        # Build kwargs based on operation
        kwargs = {}
        
        if operation == "caption":
            kwargs["detail_level"] = ctx.params.get("detail_level", None)
            
        elif operation == "ocr":
            kwargs["store_region_info"] = ctx.params.get("store_region_info", False)
            
        elif operation == "detection":
            kwargs["detection_type"] = ctx.params.get("detection_type", None)
            if ctx.params.get("detection_type") == "open_vocabulary_detection":
                kwargs["text_prompt"] = ctx.params.get("text_prompt", None)
                
        elif operation == "phrase_grounding":
            input_type = ctx.params.get("caption_input", "caption")
            if input_type == "caption_field":
                kwargs["caption_field"] = ctx.params.get("caption_field")
            else:
                kwargs["caption"] = ctx.params.get("caption")
                
        elif operation == "segmentation":
            input_type = ctx.params.get("expression_input", "expression")
            if input_type == "expression_field":
                kwargs["expression_field"] = ctx.params.get("expression_field")
            else:
                kwargs["expression"] = ctx.params.get("expression")
        
        run_florence2_model(
            view,
            operation=operation,
            output_field=output_field,
            model_path=model_path,
            **kwargs
        )
        
        dataset.add_dynamic_sample_fields()
        ctx.ops.reload_dataset()
    
    # Convenience methods for each operation type
    def caption(
        self,
        sample_collection,
        output_field,
        detail_level="basic",
        model_path=DEFAULT_MODEL_PATH,
        delegate=False
    ):
        """Generate captions using Florence-2.
        
        Args:
            sample_collection: FiftyOne dataset or view
            output_field: Field to store results in
            detail_level: Level of caption detail
            model_path: Path to Florence-2 model
            delegate: Whether to delegate execution
            
        Returns:
            Operation result
        """
        return self.__call__(
            sample_collection,
            "caption",
            output_field,
            model_path,
            delegate,
            detail_level=detail_level
        )
    
    def ocr(
        self,
        sample_collection,
        output_field,
        store_region_info=False,
        model_path=DEFAULT_MODEL_PATH,
        delegate=False
    ):
        """Perform OCR using Florence-2.
        
        Args:
            sample_collection: FiftyOne dataset or view
            output_field: Field to store results in
            store_region_info: Whether to include region information
            model_path: Path to Florence-2 model
            delegate: Whether to delegate execution
            
        Returns:
            Operation result
        """
        return self.__call__(
            sample_collection,
            "ocr",
            output_field,
            model_path,
            delegate,
            store_region_info=store_region_info
        )
    
    def detect(
        self,
        sample_collection,
        output_field,
        detection_type="detection",
        text_prompt=None,
        model_path=DEFAULT_MODEL_PATH,
        delegate=False
    ):
        """Detect objects using Florence-2.
        
        Args:
            sample_collection: FiftyOne dataset or view
            output_field: Field to store results in
            detection_type: Type of detection to perform
            text_prompt: Text prompt for what to detect
            model_path: Path to Florence-2 model
            delegate: Whether to delegate execution
            
        Returns:
            Operation result
        """
        kwargs = {"detection_type": detection_type}
        if text_prompt is not None:
            kwargs["text_prompt"] = text_prompt
            
        return self.__call__(
            sample_collection,
            "detection",
            output_field,
            model_path,
            delegate,
            **kwargs
        )
    
    def ground_phrases(
        self,
        sample_collection,
        output_field,
        caption=None,
        caption_field=None,
        model_path=DEFAULT_MODEL_PATH,
        delegate=False
    ):
        """Ground phrases in captions using Florence-2.
        
        Args:
            sample_collection: FiftyOne dataset or view
            output_field: Field to store results in
            caption: Direct caption text
            caption_field: Field containing captions
            model_path: Path to Florence-2 model
            delegate: Whether to delegate execution
            
        Returns:
            Operation result
        """
        if caption is None and caption_field is None:
            raise ValueError("Either caption or caption_field must be provided")
            
        kwargs = {}
        if caption is not None:
            kwargs["caption"] = caption
        if caption_field is not None:
            kwargs["caption_field"] = caption_field
            
        return self.__call__(
            sample_collection,
            "phrase_grounding",
            output_field,
            model_path,
            delegate,
            **kwargs
        )
    
    def segment(
        self,
        sample_collection,
        output_field,
        expression=None,
        expression_field=None,
        model_path=DEFAULT_MODEL_PATH,
        delegate=False
    ):
        """Perform referring expression segmentation using Florence-2.
        
        Args:
            sample_collection: FiftyOne dataset or view
            output_field: Field to store results in
            expression: Direct referring expression
            expression_field: Field containing expressions
            model_path: Path to Florence-2 model
            delegate: Whether to delegate execution
            
        Returns:
            Operation result
        """
        if expression is None and expression_field is None:
            raise ValueError("Either expression or expression_field must be provided")
            
        kwargs = {}
        if expression is not None:
            kwargs["expression"] = expression
        if expression_field is not None:
            kwargs["expression_field"] = expression_field
            
        return self.__call__(
            sample_collection,
            "segmentation",
            output_field,
            model_path,
            delegate,
            **kwargs
        )
    
    def __call__(
        self,
        sample_collection,
        operation,
        output_field,
        model_path=DEFAULT_MODEL_PATH,
        delegate=False,
        **kwargs
    ):
        """Base method for calling Florence-2 operations.
        
        Args:
            sample_collection: FiftyOne dataset or view
            operation: Type of operation to perform
            output_field: Field to store results in
            model_path: Path to Florence-2 model
            delegate: Whether to delegate execution
            **kwargs: Operation-specific parameters
            
        Returns:
            Operation result
        """
        return _handle_calling(
            self.uri,
            sample_collection,
            model_path,
            operation,
            output_field,
            delegate,
            **kwargs
        )

def register(plugin):
    """Register operators with the plugin."""
    # Register individual task operators
    plugin.register(CaptionWithFlorence2())
    plugin.register(OCRWithFlorence2())
    plugin.register(DetectWithFlorence2())
    plugin.register(CaptionToPhraseGroundingWithFlorence2())
    plugin.register(ReferringExpressionSegmentationWithFlorence2())
    
    # Register combined operator
    plugin.register(Florence2Operator())

# For compatibility with previous versions
def florence2_activator():
    """Check if required dependencies are installed."""
    from importlib.util import find_spec
    return (find_spec("transformers") is not None and 
            find_spec("torch") is not None and
            find_spec("PIL") is not None)