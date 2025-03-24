# caption_operator.py
import os
os.environ['FIFTYONE_ALLOW_LEGACY_ORCHESTRATORS'] = 'true'
import fiftyone as fo
import fiftyone.operators as foo
from fiftyone.operators import types

from .florence2 import run_florence2_model

from .utils import  _model_choice_inputs, _execution_mode, _handle_calling

class CaptionWithFlorence2(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="caption_with_florence2",
            label="Caption with Florence-2",
            description="Generate image captions using Florence-2",
            icon="/assets/santa-maria-del-fiore-svgrepo-com.svg",
            dynamic=True,
        )
    
    def resolve_input(self, ctx):
        inputs = types.Object()
        
        # Model choice inputs
        _model_choice_inputs(ctx, inputs)
        
        # Detail level dropdown
        detail_level_dropdown = types.RadioGroup()
        detail_level_dropdown.add_choice("basic", label="Basic caption")
        detail_level_dropdown.add_choice("detailed", label="Detailed caption")
        detail_level_dropdown.add_choice("more_detailed", label="More detailed caption")
        
        inputs.enum(
            "detail_level",
            values=detail_level_dropdown.values(),
            view=detail_level_dropdown,
            label="Caption Detail Level",
            description="Choose the level of detail for the caption"
        )
        
        # Output field
        inputs.str(
            "output_field",
            default="florence2_caption",
            required=True,
            label="Output Field",
            description="Name of the field to store the caption"
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
        detail_level = ctx.params.get("detail_level")
        output_field = ctx.params.get("output_field")
        
        # Execute model
        run_florence2_model(
            dataset=view,
            operation="caption",
            output_field=output_field,
            model_path=model_path,
            detail_level=detail_level
        )
        
        ctx.ops.reload_dataset()
        
    def __call__(
        self,
        sample_collection,
        model_path="microsoft/Florence-2-base-ft",
        detail_level="basic",
        output_field="florence2_caption",
        delegate=False
    ):
        return _handle_calling(
            self.uri,
            sample_collection,
            operation="caption",
            output_field=output_field,
            delegate=delegate,
            model_path=model_path,
            detail_level=detail_level
        )