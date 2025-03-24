# detection_operator.py
import os
os.environ['FIFTYONE_ALLOW_LEGACY_ORCHESTRATORS'] = 'true'
import fiftyone as fo
import fiftyone.operators as foo
from fiftyone.operators import types

from .florence2 import run_florence2_model

from .utils import  _model_choice_inputs, _execution_mode, _handle_calling

def _detection_label_field_inputs(inputs):
    inputs.str(
        "label_field",
        label="Label field",
        description="The field in which to store the detection results",
    )


def _detection_task_choice_inputs(ctx, inputs):
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
        )

class DetectWithFlorence2(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="detect_with_florence2",
            label="Detect with Florence-2",
            description="Detect objects in images using Florence-2",
            icon="/assets/santa-maria-del-fiore-svgrepo-com.svg",
            dynamic=True,
        )
    
    def resolve_input(self, ctx):
        inputs = types.Object()
        
        # Model choice inputs
        _model_choice_inputs(ctx, inputs)
        
        _detection_task_choice_inputs(ctx, inputs)

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