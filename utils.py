import os
os.environ['FIFTYONE_ALLOW_LEGACY_ORCHESTRATORS'] = 'true'

import fiftyone as fo
import fiftyone.operators as foo
from fiftyone.operators import types

from .florence2 import run_florence2_model, DEFAULT_MODEL_PATH

# Common UI utilities
def _model_choice_inputs(ctx, inputs):
    model_paths = [
        "microsoft/Florence-2-base",
        "microsoft/Florence-2-large",
        "microsoft/Florence-2-base-ft",
        "microsoft/Florence-2-large-ft",
    ]

    radio_group = types.RadioGroup()
    for model_path in model_paths:
        radio_group.add_choice(model_path, label=model_path)

    inputs.enum(
        "model_path",
        radio_group.values(),
        label="Model path",
        description="The model checkpoint to use for the operation",
        required=False,
        view=types.DropdownView(),
    )

    _model_download_check_inputs(ctx, inputs)

def _model_download_check_inputs(ctx, inputs):
    model_choice = ctx.params.get("model_path", None)
    if model_choice is None:
        return

    base_path = "~/.cache/huggingface/hub/models--"
    model_path_formatted = base_path + model_choice.replace("/", "--")
    model_dir = os.path.expanduser(model_path_formatted)

    if not os.path.exists(model_dir):
        description = (
            f"Model {model_choice} has not been downloaded. The model will be "
            "downloaded automatically the first time you run this operation."
            "Please be aware that this may take some time."
        )
        inputs.view(
            "model_download_warning",
            types.Warning(
                label="Model not downloaded", description=description
            ),
        )

def _execution_mode(ctx, inputs):
    delegate = ctx.params.get("delegate", False)

    if delegate:
        description = "Uncheck this box to execute the operation immediately"
    else:
        description = "Check this box to delegate execution of this task"

    inputs.bool(
        "delegate",
        default=False,
        required=True,
        label="Delegate execution?",
        description=description,
        view=types.CheckboxView(),
    )

    if delegate:
        inputs.view(
            "notice",
            types.Notice(
                label=(
                    "You've chosen delegated execution. Note that you must "
                    "have a delegated operation service running in order for "
                    "this task to be processed. See "
                    "https://docs.voxel51.com/plugins/index.html#operators "
                    "for more information"
                )
            ),
        )

def _handle_calling(
        uri, 
        sample_collection, 
        model_path,
        operation,
        output_field,
        delegate=False,
        **kwargs
        ):
    """Helper function to handle operator calling via SDK."""
    ctx = dict(dataset=sample_collection)

    params = dict(
        model_path=model_path,
        operation=operation,
        output_field=output_field,
        delegate=delegate,
        **kwargs
        )
    return foo.execute_operator(uri, ctx, params=params)
