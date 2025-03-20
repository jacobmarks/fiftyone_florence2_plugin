import fiftyone as fo
import fiftyone.operators as foo
from fiftyone.operators import types

from .florence2 import  DEFAULT_MODEL_PATH

from .utils import _handle_calling, _BaseFlorence2Operator


class ReferringExpressionSegmentationWithFlorence2(_BaseFlorence2Operator):
    """Operator for performing referring expression segmentation with Florence-2."""
    
    def __init__(self):
        self.operation = "segmentation"
        self.operation_label = "perform referring expression segmentation with Florence-2"
        self.form_label = "Referring expression segmentation with Florence-2"
        self.form_description = "Perform referring expression segmentation with Florence-2"
    
    def _add_operation_inputs(self, ctx, inputs):
        # Expression source
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
            
        # Segmentation field
        inputs.str(
            "segmentation_field",
            label="Segmentation field",
            description="The field in which to store the segmentation results",
            required=False,
        )
    
    def _get_operation_kwargs(self, ctx):
        input_type = ctx.params.get("expression_input", "expression")
        
        if input_type == "expression_field":
            return {
                "expression_field": ctx.params.get("expression_field")
            }
        else:
            return {
                "expression": ctx.params.get("expression")
            }
            
    def __call__(
        self, 
        sample_collection,
        output_field=None,
        expression=None,  # Explicit parameter
        expression_field=None,  # Explicit parameter
        model_path=DEFAULT_MODEL_PATH,
        delegate=False
    ):
        """Perform referring expression segmentation with Florence-2.
        
        Args:
            sample_collection: FiftyOne dataset or view to process
            output_field: Field to store segmentation results in
            expression: Direct referring expression to use (provide either this or expression_field)
            expression_field: Field containing expressions to use (provide either this or expression)
            model_path: Path to Florence-2 model
            delegate: Whether to delegate execution
            
        Returns:
            The operation result
            
        Raises:
            ValueError: If neither expression nor expression_field is provided
        """
        if expression is None and expression_field is None:
            raise ValueError("Either expression or expression_field must be provided")
            
        kwargs = {}
        if expression is not None:
            kwargs["expression"] = expression
        if expression_field is not None:
            kwargs["expression_field"] = expression_field
            
        return _handle_calling(
            self.uri,
            sample_collection,
            model_path,
            self.operation,
            output_field,
            delegate,
            **kwargs
        )
