"""GPT-4o plugin.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import os
from unittest.mock import patch

from PIL import Image

import fiftyone as fo
import fiftyone.operators as foo
from fiftyone.operators import types

from transformers.dynamic_module_utils import get_imports
from transformers import AutoModelForCausalLM, AutoProcessor

DEFAULT_MODEL_PATH = "microsoft/Florence-2-base"


def fixed_get_imports(filename) -> list[str]:
    if not str(filename).endswith("modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    imports.remove("flash_attn")
    return imports


def _get_model_and_processor(model_path):
    with patch(
        "transformers.dynamic_module_utils.get_imports", fixed_get_imports
    ):  # workaround for unnecessary flash_attn requirement
        model = AutoModelForCausalLM.from_pretrained(
            model_path, attn_implementation="sdpa", trust_remote_code=True
        )

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    return model, processor


def _generate_and_parse(
    model, processor, task, image, text_input=None, max_new_tokens=1024, num_beams=3
):
    text = task
    if text_input is not None:
        text = text_input
    inputs = processor(text=text, images=image, return_tensors="pt")

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        do_sample=False,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    parsed_answer = processor.post_process_generation(
        generated_text, task=task, image_size=(image.width, image.height)
    )

    return parsed_answer


def _task_to_field_name(task):
    fmt_task = task.lower().replace("<", "").replace(">", "").replace(" ", "_")
    return f"florence2_{fmt_task}"


def _convert_bbox(bbox, width, height):
    if len(bbox) == 4:
        return [
            bbox[0] / width,
            bbox[1] / height,
            (bbox[2] - bbox[0]) / width,
            (bbox[3] - bbox[1]) / height,
        ]
    else:
        ## quad_boxes
        x1, y1, x2, y2, x3, y3, x4, y4 = bbox
        x_min = min(x1, x2, x3, x4)
        x_max = max(x1, x2, x3, x4)
        y_min = min(y1, y2, y3, y4)
        y_max = max(y1, y2, y3, y4)

        return [
            x_min / width,
            y_min / height,
            (x_max - x_min) / width,
            (y_max - y_min) / height,
        ]


def _extract_detections(parsed_answer, task, image):
    label_key = "bboxes_labels" if task == "<OPEN_VOCABULARY_DETECTION>" else "labels"
    bbox_key = "quad_boxes" if task == "<OCR_WITH_REGION>" else "bboxes"
    bboxes = parsed_answer[task][bbox_key]
    labels = parsed_answer[task][label_key]
    dets = []
    for i, (bbox, label) in enumerate(zip(bboxes, labels)):
        dets.append(
            fo.Detection(
                label=label if label else f"object_{i+1}",
                bounding_box=_convert_bbox(bbox, image.width, image.height),
            )
        )
    return fo.Detections(detections=dets)


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
            types.Warning(label="Model not downloaded", description=description),
        )


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


def detect_with_florence2(
    sample_collection,
    detection_type=None,
    model_path="microsoft/Florence-2-base",
    label_field=None,
    text_prompt=None,
):
    if detection_type is None or detection_type == "detection":
        task = "<OD>"
    elif detection_type == "dense_region_caption":
        task = "<DENSE_REGION_CAPTION>"
    elif detection_type == "region_proposal":
        task = "<REGION_PROPOSAL>"
    elif detection_type == "open_vocabulary_detection":
        task = "<OPEN_VOCABULARY_DETECTION>"
    elif detection_type == "ocr_with_region":
        task = "<OCR_WITH_REGION>"

    if label_field is None:
        label_field = _task_to_field_name(task)

    model, processor = _get_model_and_processor(model_path)
    for sample in sample_collection.iter_samples(autosave=True):
        image = Image.open(sample.filepath).convert("RGB")
        parsed_answer = _generate_and_parse(
            model, processor, task, image, text_input=text_prompt
        )
        detections = _extract_detections(parsed_answer, task, image)
        sample[label_field] = detections


def caption_with_florence2(
    sample_collection,
    detail_level=None,
    model_path="microsoft/Florence-2-base",
    caption_field=None,
):
    if detail_level is None or detail_level == "basic":
        task = "<CAPTION>"
    elif detail_level == "detailed":
        task = "<DETAILED_CAPTION>"
    elif detail_level == "more_detailed":
        task = "<MORE_DETAILED_CAPTION>"

    if caption_field is None:
        caption_field = _task_to_field_name(task)

    model, processor = _get_model_and_processor(model_path)
    for sample in sample_collection.iter_samples(autosave=True):
        image = Image.open(sample.filepath).convert("RGB")
        parsed_answer = _generate_and_parse(model, processor, task, image)
        sample[caption_field] = parsed_answer[task]


def caption_to_phrase_grounding_with_florence2(
    sample_collection,
    model_path="microsoft/Florence-2-base",
    caption_field=None,
    caption=None,
    label_field=None,
):

    def _resolve_caption(sample, caption_field, caption):
        if caption_field is not None:
            return sample[caption_field]
        elif caption is not None:
            return caption
        else:
            raise ValueError("Either `caption_field` or `caption` must be provided")

    task = "<CAPTION_TO_PHRASE_GROUNDING>"
    if label_field is None:
        label_field = _task_to_field_name(task)

    model, processor = _get_model_and_processor(model_path)
    for sample in sample_collection.iter_samples(autosave=True):
        image = Image.open(sample.filepath).convert("RGB")
        sample_caption = _resolve_caption(sample, caption_field, caption)
        parsed_answer = _generate_and_parse(
            model, processor, task, image, text_input=sample_caption
        )
        detections = _extract_detections(parsed_answer, task, image)
        sample[label_field] = detections


def referring_expression_segmentation_with_florence2(
    sample_collection,
    model_path="microsoft/Florence-2-base",
    text_input=None,
    text_field=None,
    label_field=None,
):
    task = "<REFERRING_EXPRESSION_SEGMENTATION>"

    if label_field is None:
        label_field = _task_to_field_name(task)

    def _resolve_text_input(sample, text_field, text_input):
        if text_field is not None:
            return sample[text_field]
        elif text_input is not None:
            return text_input
        else:
            raise ValueError("Either `text_field` or `text_input` must be provided")

    model, processor = _get_model_and_processor(model_path)
    for sample in sample_collection.iter_samples(autosave=True):
        image = Image.open(sample.filepath).convert("RGB")
        _text = _resolve_text_input(sample, text_field, text_input)
        parsed_answer = _generate_and_parse(
            model, processor, task, image, text_input=_text
        )
        polygons = parsed_answer[task]["polygons"]
        if not polygons:
            sample[label_field] = None
            continue

        pls = []

        for k, polygon in enumerate(polygons):
            _polygon = polygon[0]
            x_points = [p for i, p in enumerate(_polygon) if i % 2 == 0]
            y_points = [p for i, p in enumerate(_polygon) if i % 2 != 0]
            x_points = [x / image.width for x in x_points]
            y_points = [y / image.height for y in y_points]

            xy_points = []
            curr_x = x_points[0]
            curr_y = y_points[0]
            xy_points.append((curr_x, curr_y))
            for i in range(1, len(x_points)):
                curr_x = x_points[i]
                xy_points.append((curr_x, curr_y))
                curr_y = y_points[i]
                xy_points.append((curr_x, curr_y))

            ## handle last point
            xy_points.append((x_points[0], curr_y))

            pls.append(
                fo.Polyline(
                    points=[xy_points], label=f"object_{k+1}", filled=True, closed=True
                )
            )

        sample[label_field] = fo.Polylines(polylines=pls)


def ocr_with_florence2(
    sample_collection, model_path="microsoft/Florence-2-base", label_field=None
):
    task = "<OCR>"

    if label_field is None:
        label_field = _task_to_field_name(task)

    model, processor = _get_model_and_processor(model_path)
    for sample in sample_collection.iter_samples(autosave=True):
        image = Image.open(sample.filepath).convert("RGB")
        parsed_answer = _generate_and_parse(model, processor, task, image)
        sample[label_field] = parsed_answer[task]


def _detail_level_inputs(inputs):
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


def _caption_field_inputs(inputs):
    inputs.str(
        "caption_field",
        label="Caption field",
        description="The field in which to store the generated caption",
        required=False,
    )


class CaptionWithFlorence2(foo.Operator):
    @property
    def config(self):
        _config = foo.OperatorConfig(
            name="caption_with_florence2",
            label="Florence2: caption images with Florence-2",
            dynamic=True,
        )
        return _config

    def resolve_delegation(self, ctx):
        return ctx.params.get("delegate", False)

    def resolve_input(self, ctx):
        inputs = types.Object()
        form_view = types.View(
            label="Caption with Florence-2",
            description=(
                "Generate a caption for each image in the dataset using" "Florence-2"
            ),
        )
        _model_choice_inputs(ctx, inputs)
        _caption_field_inputs(inputs)
        _detail_level_inputs(inputs)
        _execution_mode(ctx, inputs)
        inputs.view_target(ctx)
        return types.Property(inputs, view=form_view)

    def execute(self, ctx):
        dataset = ctx.dataset
        view = ctx.target_view()
        model_path = ctx.params.get("model_path", DEFAULT_MODEL_PATH)
        caption_field = ctx.params.get("caption_field", None)
        detail_level = ctx.params.get("detail_level", None)
        caption_with_florence2(
            view,
            detail_level=detail_level,
            caption_field=caption_field,
            model_path=model_path,
        )
        dataset.add_dynamic_sample_fields()
        ctx.ops.reload_dataset()


def _store_region_info_inputs(inputs):
    inputs.bool(
        "store_region_info",
        default=False,
        label="Store region information?",
        description="Check this box to represent OCR detections with `fo.Detections` objects, which include bounding box information",
        view=types.CheckboxView(),
    )


def _ocr_label_field_inputs(inputs):
    inputs.str(
        "label_field",
        label="Label field",
        description="The field in which to store the OCR results",
        required=False,
    )


class OCRWithFlorence2(foo.Operator):
    @property
    def config(self):
        _config = foo.OperatorConfig(
            name="ocr_with_florence2",
            label="Florence2: perform OCR on images with Florence-2",
            dynamic=True,
        )
        return _config

    def resolve_delegation(self, ctx):
        return ctx.params.get("delegate", False)

    def resolve_input(self, ctx):
        inputs = types.Object()
        form_view = types.View(
            label="OCR with Florence-2",
            description=("Perform OCR on images using Florence-2"),
        )
        _store_region_info_inputs(inputs)
        _model_choice_inputs(ctx, inputs)
        _ocr_label_field_inputs(inputs)
        _execution_mode(ctx, inputs)
        inputs.view_target(ctx)
        return types.Property(inputs, view=form_view)

    def execute(self, ctx):
        dataset = ctx.dataset
        view = ctx.target_view()
        model_path = ctx.params.get("model_path", DEFAULT_MODEL_PATH)
        label_field = ctx.params.get("label_field", None)

        store_region_info = ctx.params.get("store_region_info", False)
        if store_region_info:
            detect_with_florence2(
                view,
                detection_type="ocr_with_region",
                model_path=model_path,
                label_field=label_field,
            )
        else:
            ocr_with_florence2(view, model_path=model_path, label_field=label_field)

        dataset.add_dynamic_sample_fields()
        ctx.ops.reload_dataset()


def _detection_label_field_inputs(inputs):
    inputs.str(
        "label_field",
        label="Label field",
        description="The field in which to store the detection results",
        required=False,
    )


def _detection_task_choice_inputs(ctx, inputs):
    detection_task_choices = [
        "detection",
        "dense_region_caption",
        "region_proposal",
        "open_vocabulary_detection",
    ]

    radio_group = types.RadioGroup()
    for choice in detection_task_choices:
        radio_group.add_choice(choice, label=choice)

    inputs.enum(
        "detection_type",
        radio_group.values(),
        label="Detection type",
        description="The type of detection to perform",
        required=False,
        view=types.DropdownView(),
    )

    detection_task = ctx.params.get("detection_type", None)
    if detection_task is None:
        return

    if detection_task == "open_vocabulary_detection":
        inputs.str(
            "text_prompt",
            label="Text prompt",
            description="What do you want to detect?",
            required=True,
        )


class DetectWithFlorence2(foo.Operator):
    @property
    def config(self):
        _config = foo.OperatorConfig(
            name="detect_with_florence2",
            label="Florence2: detect objects in images with Florence-2",
            dynamic=True,
        )
        return _config

    def resolve_delegation(self, ctx):
        return ctx.params.get("delegate", False)

    def resolve_input(self, ctx):
        inputs = types.Object()
        form_view = types.View(
            label="Detect with Florence-2",
            description=("Detect objects in images using Florence-2"),
        )
        _model_choice_inputs(ctx, inputs)
        _detection_task_choice_inputs(ctx, inputs)
        _detection_label_field_inputs(inputs)
        _execution_mode(ctx, inputs)
        inputs.view_target(ctx)
        return types.Property(inputs, view=form_view)

    def execute(self, ctx):
        dataset = ctx.dataset
        view = ctx.target_view()
        model_path = ctx.params.get("model_path", DEFAULT_MODEL_PATH)
        label_field = ctx.params.get("label_field", None)
        detection_type = ctx.params.get("detection_type", None)
        text_prompt = ctx.params.get("text_prompt", None)

        detect_with_florence2(
            view,
            detection_type=detection_type,
            model_path=model_path,
            label_field=label_field,
            text_prompt=text_prompt,
        )

        dataset.add_dynamic_sample_fields()
        ctx.ops.reload_dataset()


def _caption_inputs(ctx, inputs):
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


class CaptionToPhraseGroundingWithFlorence2(foo.Operator):
    @property
    def config(self):
        _config = foo.OperatorConfig(
            name="caption_to_phrase_grounding_with_florence2",
            label="Florence2: Ground phrases in captions with Florence-2",
            dynamic=True,
        )
        return _config

    def resolve_delegation(self, ctx):
        return ctx.params.get("delegate", False)

    def resolve_input(self, ctx):
        inputs = types.Object()
        form_view = types.View(
            label="Caption to phrase grounding with Florence-2",
            description=("Ground phrases in captions with Florence-2"),
        )
        _model_choice_inputs(ctx, inputs)
        _caption_inputs(ctx, inputs)
        _detection_label_field_inputs(inputs)
        _execution_mode(ctx, inputs)
        inputs.view_target(ctx)
        return types.Property(inputs, view=form_view)

    def execute(self, ctx):
        dataset = ctx.dataset
        view = ctx.target_view()
        model_path = ctx.params.get("model_path", DEFAULT_MODEL_PATH)
        caption_field = ctx.params.get("caption_field", None)
        caption = ctx.params.get("caption", None)
        label_field = ctx.params.get("label_field", None)

        caption_to_phrase_grounding_with_florence2(
            view,
            model_path=model_path,
            caption_field=caption_field,
            caption=caption,
            label_field=label_field,
        )

        dataset.add_dynamic_sample_fields()
        ctx.ops.reload_dataset()



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
        required=True,
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


def _polylines_label_field_inputs(inputs):
    inputs.str(
        "label_field",
        label="Label field",
        description="The field in which to store the segmentation results",
        required=False,
    )

class ReferringExpressionSegmentationWithFlorence2(foo.Operator):
    @property
    def config(self):
        _config = foo.OperatorConfig(
            name="referring_expression_segmentation_with_florence2",
            label="Florence2: perform referring expression segmentation with Florence-2",
            dynamic=True,
        )
        return _config

    def resolve_delegation(self, ctx):
        return ctx.params.get("delegate", False)

    def resolve_input(self, ctx):
        inputs = types.Object()
        form_view = types.View(
            label="Referring expression segmentation with Florence-2",
            description=("Perform referring expression segmentation with Florence-2"),
        )
        _model_choice_inputs(ctx, inputs)
        _referring_expression_inputs(ctx, inputs)
        _polylines_label_field_inputs(inputs)
        _execution_mode(ctx, inputs)
        inputs.view_target(ctx)
        return types.Property(inputs, view=form_view)

    def execute(self, ctx):
        dataset = ctx.dataset
        view = ctx.target_view()
        model_path = ctx.params.get("model_path", DEFAULT_MODEL_PATH)
        text_field = ctx.params.get("expression_field", None)
        text_input = ctx.params.get("expression", None)
        label_field = ctx.params.get("label_field", None)

        referring_expression_segmentation_with_florence2(
            view,
            model_path=model_path,
            text_field=text_field,
            text_input=text_input,
            label_field=label_field,
        )

        dataset.add_dynamic_sample_fields()
        ctx.ops.reload_dataset()


def register(plugin):
    plugin.register(CaptionWithFlorence2)
    plugin.register(OCRWithFlorence2)
    plugin.register(DetectWithFlorence2)
    plugin.register(CaptionToPhraseGroundingWithFlorence2)
    plugin.register(ReferringExpressionSegmentationWithFlorence2)
