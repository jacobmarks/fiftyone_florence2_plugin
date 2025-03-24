# ocr_operator.py
import os
os.environ['FIFTYONE_ALLOW_LEGACY_ORCHESTRATORS'] = 'true'
import fiftyone as fo
import fiftyone.operators as foo
from fiftyone.operators import types

from .florence2 import run_florence2_model

from .utils import  _model_choice_inputs, _execution_mode, _handle_calling

class OCRWithFlorence2(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="ocr_with_florence2",
            label="OCR with Florence-2",
            description="Perform optical character recognition using Florence-2",
            icon="/assets/santa-maria-del-fiore-svgrepo-com.svg",
            dynamic=True,
        )
    
    def resolve_input(self, ctx):
        inputs = types.Object()
        
        # Model choice inputs
        _model_choice_inputs(ctx, inputs)
        
        # Store region info option
        inputs.bool(
            "store_region_info",
            default=False,
            required=True,
            label="Include Region Information",
            description="Include bounding box information for detected text",
            view=types.CheckboxView(),
        )
        
        # Output field
        inputs.str(
            "output_field",
            default="florence2_ocr",
            required=True,
            label="Output Field",
            description="Name of the field to store the OCR results"
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
        store_region_info = ctx.params.get("store_region_info", False)
        output_field = ctx.params.get("output_field")
        
        # Execute model
        run_florence2_model(
            dataset=view,
            operation="ocr",
            output_field=output_field,
            model_path=model_path,
            store_region_info=store_region_info
        )
        
        ctx.ops.reload_dataset()
        
    def __call__(
        self,
        sample_collection,
        model_path="microsoft/Florence-2-base-ft",
        store_region_info=False,
        output_field="florence2_ocr",
        delegate=False
    ):
        return _handle_calling(
            self.uri,
            sample_collection,
            operation="ocr",
            output_field=output_field,
            delegate=delegate,
            model_path=model_path,
            store_region_info=store_region_info
        )