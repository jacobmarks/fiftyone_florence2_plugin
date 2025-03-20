import fiftyone as fo
import fiftyone.operators as foo
from fiftyone.operators import types

from .florence2 import  DEFAULT_MODEL_PATH

from .utils import _handle_calling, _BaseFlorence2Operator

# Specific operator classes
class CaptionWithFlorence2(_BaseFlorence2Operator):
    """Operator for captioning images with Florence-2."""
    
    @property
    def config(self):
        _config = foo.OperatorConfig(
            name="caption_with_florence2",
            label="Florence2: caption images with Florence-2",
            dynamic=True,
        )
        return _config
    
    def _add_operation_inputs(self, ctx, inputs):
        # Caption field
        inputs.str(
            "caption_field",
            label="Caption field",
            description="The field in which to store the generated caption",
            required=False,
        )
        
        # Detail level
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
    
    def _get_operation_kwargs(self, ctx):
        return {
            "detail_level": ctx.params.get("detail_level", None)
        }
        
    def __call__(
        self, 
        sample_collection,
        output_field=None,
        detail_level="basic",  # Explicit parameter 
        model_path=DEFAULT_MODEL_PATH,
        delegate=False
    ):
        """Generate captions for images using Florence-2.
        
        Args:
            sample_collection: FiftyOne dataset or view to process
            output_field: Field to store captions in
            detail_level: Level of caption detail. Options: "basic", "detailed", "more_detailed"
            model_path: Path to Florence-2 model
            delegate: Whether to delegate execution
            
        Returns:
            The operation result
        """
        return _handle_calling(
            self.uri,
            sample_collection,
            model_path,
            self.operation,
            output_field,
            delegate,
            detail_level=detail_level
        )
