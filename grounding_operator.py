import fiftyone as fo
import fiftyone.operators as foo
from fiftyone.operators import types

from .florence2 import  DEFAULT_MODEL_PATH

from .utils import _handle_calling, _BaseFlorence2Operator

# Specific operator classes
class CaptionToPhraseGroundingWithFlorence2(_BaseFlorence2Operator):
    """Operator for grounding phrases in captions with Florence-2."""
    
    def __init__(self):
        self.operation = "phrase_grounding"
        self.operation_label = "Ground phrases in captions with Florence-2"
        self.form_label = "Caption to phrase grounding with Florence-2"
        self.form_description = "Ground phrases in captions with Florence-2"
    
    def _add_operation_inputs(self, ctx, inputs):
        # Caption source
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
            
        # Phrase grounding field
        inputs.str(
            "phrase_grounding_field",
            label="Phrase grounding field",
            description="The field in which to store the grounded phrases",
            required=False,
        )
    
    def _get_operation_kwargs(self, ctx):
        input_type = ctx.params.get("caption_input", "caption")
        
        if input_type == "caption_field":
            return {
                "caption_field": ctx.params.get("caption_field")
            }
        else:
            return {
                "caption": ctx.params.get("caption")
            }
            
    def __call__(
        self, 
        sample_collection,
        output_field=None,
        caption=None,  # Explicit parameter
        caption_field=None,  # Explicit parameter
        model_path=DEFAULT_MODEL_PATH,
        delegate=False
    ):
        """Ground phrases in captions with Florence-2.
        
        Args:
            sample_collection: FiftyOne dataset or view to process
            output_field: Field to store grounding results in
            caption: Direct caption text to use (provide either this or caption_field)
            caption_field: Field containing captions to use (provide either this or caption)
            model_path: Path to Florence-2 model
            delegate: Whether to delegate execution
            
        Returns:
            The operation result
            
        Raises:
            ValueError: If neither caption nor caption_field is provided
        """
        if caption is None and caption_field is None:
            raise ValueError("Either caption or caption_field must be provided")
            
        kwargs = {}
        if caption is not None:
            kwargs["caption"] = caption
        if caption_field is not None:
            kwargs["caption_field"] = caption_field
            
        return _handle_calling(
            self.uri,
            sample_collection,
            model_path,
            self.operation,
            output_field,
            delegate,
            **kwargs
        )