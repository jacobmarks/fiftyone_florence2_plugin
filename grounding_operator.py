# grounding_operator.py
import os

import fiftyone as fo
import fiftyone.operators as foo
from fiftyone.operators import types

from .florence2 import run_florence2_model

from .utils import  _model_choice_inputs, _execution_mode, _handle_calling

class CaptionToPhraseGroundingWithFlorence2(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="caption_to_phrase_grounding_with_florence2",
            label="Caption to Phrase Grounding with Florence-2",
            description="Ground caption phrases in images using Florence-2",
            icon="/assets/icon-grounding.svg",  # Placeholder icon
            dynamic=True,
        )
    
    def resolve_input(self, ctx):
        inputs = types.Object()
        
        # Model choice inputs
        _model_choice_inputs(ctx, inputs)
        
        # Input source radio group
        input_source_radio = types.RadioGroup()
        input_source_radio.add_choice("direct", label="Direct Input")
        input_source_radio.add_choice("field", label="From Field")
        
        inputs.enum(
            "input_source",
            values=input_source_radio.values(),
            default="direct",
            view=input_source_radio,
            label="Caption Source",
            description="Choose where to get the caption from"
        )
        
        # Conditional UI based on input source
        input_source = ctx.params.get("input_source", "direct")
        
        if input_source == "direct":
            inputs.str(
                "caption",
                default="",
                required=True,
                label="Caption",
                description="Enter the caption to ground in the image"
            )
        else:  # input_source == "field"
            # Get available string fields from the dataset
            string_fields = []
            try:
                if ctx.dataset:
                    string_fields = ctx.dataset.get_field_schema(flat=True).keys()
                    string_fields = [f for f in string_fields if ctx.dataset.get_field_type(f) == "string"]
            except:
                pass
            
            # Create dropdown for fields
            field_dropdown = types.Dropdown(label="Caption Field")
            for field in string_fields:
                field_dropdown.add_choice(field, label=field)
            
            # Add field dropdown
            inputs.enum(
                "caption_field",
                values=field_dropdown.values() if field_dropdown.values() else [""],
                default=field_dropdown.values()[0] if field_dropdown.values() else "",
                view=field_dropdown,
                label="Caption Field",
                description="Choose the field containing the captions"
            )
        
        # Output field
        inputs.str(
            "output_field",
            default="florence2_grounding",
            required=True,
            label="Output Field",
            description="Name of the field to store the grounding results"
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
        input_source = ctx.params.get("input_source")
        output_field = ctx.params.get("output_field")
        
        kwargs = {}
        if input_source == "direct":
            caption = ctx.params.get("caption")
            kwargs["caption"] = caption
        else:
            caption_field = ctx.params.get("caption_field")
            kwargs["caption_field"] = caption_field
        
        # Execute model
        run_florence2_model(
            dataset=view,
            operation="phrase_grounding",
            output_field=output_field,
            model_path=model_path,
            **kwargs
        )
        
        ctx.ops.reload_dataset()
        
    def __call__(
        self,
        sample_collection,
        model_path="microsoft/Florence-2-base-ft",
        caption=None,
        caption_field=None,
        output_field="florence2_grounding",
        delegate=False
    ):
        kwargs = {}
        if caption:
            kwargs["caption"] = caption
        elif caption_field:
            kwargs["caption_field"] = caption_field
        else:
            raise ValueError("Either 'caption' or 'caption_field' must be provided")
            
        return _handle_calling(
            self.uri,
            sample_collection,
            operation="phrase_grounding",
            output_field=output_field,
            delegate=delegate,
            model_path=model_path,
            **kwargs
        )