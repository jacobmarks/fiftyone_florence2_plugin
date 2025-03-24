# detection_operator.py
import os

import fiftyone as fo
import fiftyone.operators as foo
from fiftyone.operators import types

from .florence2 import run_florence2_model

from .utils import  _model_choice_inputs, _execution_mode, _handle_calling

class DetectWithFlorence2(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="detect_with_florence2",
            label="Detect with Florence-2",
            description="Detect objects in images using Florence-2",
            icon="/assets/icon-detection.svg",  # Placeholder icon
            dynamic=True,
        )
    
    def resolve_input(self, ctx):
        inputs = types.Object()
        
        # Model choice inputs
        _model_choice_inputs(ctx, inputs)
        
        # Detection type dropdown
        detection_type_dropdown = types.Dropdown(label="Detection Type")
        detection_type_dropdown.add_choice("detection", label="Standard Object Detection")
        detection_type_dropdown.add_choice("dense_region_caption", label="Dense Region Caption")
        detection_type_dropdown.add_choice("region_proposal", label="Region Proposal")
        detection_type_dropdown.add_choice("open_vocabulary_detection", label="Open Vocabulary Detection")
        
        inputs.enum(
            "detection_type",
            values=detection_type_dropdown.values(),
            default="detection",
            view=detection_type_dropdown,
            label="Detection Type",
            description="Choose the type of detection to perform"
        )
        
        # Previous implementation used a conditional to show text_prompt only for open_vocabulary_detection,
        # but since I don't see how to access the current value of detection_type in this context, I'll always show it
        # and document when it's applicable
        
        inputs.str(
            "text_prompt",
            default="",
            required=False,
            label="Text Prompt",
            description="Text prompt for Open Vocabulary Detection (only used when Detection Type is 'Open Vocabulary Detection')"
        )
        
        # Output field
        inputs.str(
            "output_field",
            default="florence2_detections",
            required=True,
            label="Output Field",
            description="Name of the field to store the detection results"
        )
        
        # Execution mode (delegation option)
        _execution_mode(ctx, inputs)
        
        inputs.view_target(ctx)
        
        return types.Property(inputs)
    
    def resolve_delegation(self, ctx):
        return ctx.params.get("delegate", False)
    
    def execute(self, ctx):
        view = ctx.target_view()
        # Parameters
        model_path = ctx.params.get("model_path", "microsoft/Florence-2-base-ft")
        detection_type = ctx.params.get("detection_type")
        text_prompt = ctx.params.get("text_prompt")
        output_field = ctx.params.get("output_field")
        
        kwargs = {"detection_type": detection_type}
        if text_prompt and detection_type == "open_vocabulary_detection":
            kwargs["text_prompt"] = text_prompt
        
        # Execute model
        run_florence2_model(
            dataset=view,
            operation="detection",
            output_field=output_field,
            model_path=model_path,
            **kwargs
        )
        
        ctx.ops.reload_dataset()
        
    def __call__(
        self,
        sample_collection,
        model_path="microsoft/Florence-2-base-ft",
        detection_type="detection",
        text_prompt=None,
        output_field="florence2_detections",
        delegate=False
    ):
        kwargs = {"detection_type": detection_type}
        if text_prompt and detection_type == "open_vocabulary_detection":
            kwargs["text_prompt"] = text_prompt
            
        return _handle_calling(
            self.uri,
            sample_collection,
            operation="detection",
            output_field=output_field,
            delegate=delegate,
            model_path=model_path,
            **kwargs
        )