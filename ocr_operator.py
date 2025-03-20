import fiftyone as fo
import fiftyone.operators as foo
from fiftyone.operators import types

from .florence2 import  DEFAULT_MODEL_PATH

from .utils import _handle_calling, _BaseFlorence2Operator

class OCRWithFlorence2(_BaseFlorence2Operator):
    """Operator for performing OCR with Florence-2."""
    
    @property
    def config(self):
        _config = foo.OperatorConfig(
            name="ocr_with_florence2",
            label="Florence2: perform OCR on images with Florence-2",
            dynamic=True,
        )
        return _config

    
    def _add_operation_inputs(self, ctx, inputs):
        # Store region info
        inputs.bool(
            "store_region_info",
            default=False,
            label="Store region information?",
            description="Check this box to represent OCR detections with bounding boxes",
            view=types.CheckboxView(),
        )
        
        # OCR field
        inputs.str(
            "ocr_field",
            label="OCR field",
            description="The field in which to store the OCR results",
            required=False,
        )
    
    def _get_operation_kwargs(self, ctx):
        return {
            "store_region_info": ctx.params.get("store_region_info", False)
        }
        
    def __call__(
        self, 
        sample_collection,
        output_field=None,
        store_region_info=False,  # Explicit parameter
        model_path=DEFAULT_MODEL_PATH,
        delegate=False
    ):
        """Perform OCR on images using Florence-2.
        
        Args:
            sample_collection: FiftyOne dataset or view to process
            output_field: Field to store OCR results in
            store_region_info: Whether to include region information with bounding boxes
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
            store_region_info=store_region_info
        )
