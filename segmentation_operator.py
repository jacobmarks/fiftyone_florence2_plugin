# segmentation_operator.py
import os
os.environ['FIFTYONE_ALLOW_LEGACY_ORCHESTRATORS'] = 'true'
import fiftyone as fo
import fiftyone.operators as foo
from fiftyone.operators import types

from .florence2 import run_florence2_model

from .utils import  _model_choice_inputs, _execution_mode, _handle_calling

def _referring_expression_inputs(ctx, inputs):
    input_choices = ["from_field", "direct"]
    radio_group = types.RadioGroup()
    for choice in input_choices:
        radio_group.add_choice(choice, label=choice)

    inputs.enum(
        "expression_input",
        radio_group.values(),
        label="Referring expression input",
        description="The referring expression to use for segmentation",
        default="expression",
        view=types.RadioView(),
    )

    input_type = ctx.params.get("expression_input", None)
    if input_type == "from_field":
        candidate_fields = list(
            ctx.dataset.get_field_schema(ftype=fo.StringField).keys()
        )
        candidate_fields.remove("filepath")

        field_radio_group = types.RadioGroup()

        for field in candidate_fields:
            field_radio_group.add_choice(field, label=field)

        inputs.enum(
            "expression_field",
            field_radio_group.values(),
            label="Expression field",
            description="The field to use as the referring expression",
            view=types.DropdownView(),
        )

    else:
        inputs.str(
            "expression",
            label="Referring expression",
            description="The referring expression to use for segmentation",
        )

class ReferringExpressionSegmentationWithFlorence2(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="referring_expression_segmentation_with_florence2",
            label="Referring Expression Segmentation with Florence-2",
            description="Segment objects based on textual descriptions using Florence-2",
            icon="/assets/santa-maria-del-fiore-svgrepo-com.svg",
            dynamic=True,
        )
    
    def resolve_input(self, ctx):
        inputs = types.Object()
        
        # Model choice inputs
        _model_choice_inputs(ctx, inputs)
        _referring_expression_inputs(ctx, inputs)
        
        # Output field
        inputs.str(
            "output_field",
            default="florence2_segmentation",
            required=True,
            label="Output Field",
            description="Name of the field to store the segmentation results"
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
            expression = ctx.params.get("expression")
            kwargs["expression"] = expression
        else:
            expression_field = ctx.params.get("expression_field")
            kwargs["expression_field"] = expression_field
        
        # Execute model
        run_florence2_model(
            dataset=view,
            operation="segmentation",
            output_field=output_field,
            model_path=model_path,
            **kwargs
        )
        
        ctx.ops.reload_dataset()
        
    def __call__(
        self,
        sample_collection,
        model_path="microsoft/Florence-2-base-ft",
        expression=None,
        expression_field=None,
        output_field="florence2_segmentation",
        delegate=False
    ):
        kwargs = {}
        if expression:
            kwargs["expression"] = expression
        elif expression_field:
            kwargs["expression_field"] = expression_field
        else:
            raise ValueError("Either 'expression' or 'expression_field' must be provided")
            
        return _handle_calling(
            self.uri,
            sample_collection,
            operation="segmentation",
            output_field=output_field,
            delegate=delegate,
            model_path=model_path,
            **kwargs
        )