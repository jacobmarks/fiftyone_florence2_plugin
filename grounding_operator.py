# grounding_operator.py
import os
os.environ['FIFTYONE_ALLOW_LEGACY_ORCHESTRATORS'] = 'true'
import fiftyone as fo
import fiftyone.operators as foo
from fiftyone.operators import types

from .florence2 import run_florence2_model

from .utils import  _model_choice_inputs, _execution_mode, _handle_calling

def _caption_inputs(ctx, inputs):
    input_choices = ["caption_field", "caption"]
    radio_group = types.RadioGroup()
    for choice in input_choices:
        radio_group.add_choice(choice, label=choice)

    inputs.enum(
        "caption_input",
        radio_group.values(),
        label="Caption input",
        description="Please select an input to use for grounding phrases. This can be an existing Field on the Dataset or a custom caption",
        default="caption",
        view=types.RadioView(),
    )

    input_type = ctx.params.get("caption_input", None)
    if input_type == "caption_field":
        candidate_fields = list(
            ctx.dataset.get_field_schema(ftype=fo.StringField).keys()
        )
        candidate_fields.remove("filepath")

        field_radio_group = types.RadioGroup()

        for field in candidate_fields:
            field_radio_group.add_choice(field, label=field)

        inputs.enum(
            "caption_field",
            field_radio_group.values(),
            label="Caption field",
            description="The field to use as the caption",
            view=types.DropdownView(),
        )

    else:
        inputs.str(
            "caption",
            label="Caption",
            description="The caption to use for grounding phrases",
        )

class CaptionToPhraseGroundingWithFlorence2(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="caption_to_phrase_grounding_with_florence2",
            label="Caption to Phrase Grounding with Florence-2",
            description="Ground caption phrases in images using Florence-2",
            icon="/assets/santa-maria-del-fiore-svgrepo-com.svg",
            dynamic=True,
        )
    
    def resolve_input(self, ctx):
        inputs = types.Object()
        
        # Model choice inputs
        _model_choice_inputs(ctx, inputs)

        _caption_inputs(ctx, inputs)

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
        caption_input = ctx.params.get("caption_input") 
        output_field = ctx.params.get("output_field")
        
        kwargs = {}
        if caption_input == "direct":  # Changed from input_source to caption_input
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