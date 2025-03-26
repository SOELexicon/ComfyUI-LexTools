# ComfyUI-LexTools

ComfyUI-LexTools is a Python-based image processing and analysis toolkit that uses machine learning models for semantic image segmentation, image scoring, and image captioning. The toolkit includes three primary components:

1. **ImageProcessingNode.py** - Implements various image processing nodes such as:
   - `ImageAspectPadNode`: Expands the image to meet a specific aspect ratio. This node is useful for maintaining the aspect ratio when processing images.
      - _Inputs_:
         - Required: `image` (IMAGE), `aspect_ratio` (RATIO), `invert_ratio` (BOOLEAN), `feathering` (INT), `left_padding` (INT), `right_padding` (INT), `top_padding` (INT), `bottom_padding` (INT)
         - Optional: `show_on_node` (INT)
      - _Output_: Expanded Image.
   - `ImageScaleToMin`: Calculates the value needed to rescale an image's smallest dimension to 512. This node is useful for scaling images down to 512 or up to 512 for faster processing. It ensures that at least one dimension (width or height) is 512 pixels.
      - _Input_: `image` (IMAGE)
      - _Output_: Scale value.
   - `ImageRankingNode`: Ranks the images based on specific criteria.
      - _Input_: `score` (INT), `prompt` (STRING), `image_path` (STRING), `json_file_path` (STRING)
      - _Output_: Ranked images.
   - `ImageFilterByIntScoreNode` and `ImageFilterByFloatScoreNode`: Filter images based on a threshold score. Currently, these nodes may throw errors if the following node in the sequence does not handle blank outputs.
      - _Input_: `score` (INT for `ImageFilterByIntScoreNode` and FLOAT for `ImageFilterByFloatScoreNode`), `threshold` (FLOAT), `image` (IMAGE)
      - _Output_: Filtered images.
   - `ImageQualityScoreNode`: Calculates a quality score for the image.
      - _Input_: `aesthetic_score` (INT), `image_score_good` (INT), `image_score_bad` (INT), `ai_score_artificial` (INT), `ai_score_human` (INT), `weight_good_score` (INT), `weight_aesthetic_score` (INT), `weight_bad_score` (INT), `weight_AIDetection` (INT), `MultiplyScoreBy` (INT), `show_on_node` (INT), `weight_HumanDetection` (INT)
      - _Output_: Quality score.
   - `ScoreConverterNode`: Converts the score to different data types.
      - _Input_: `score` (SCORE)
      - _Output_: Converted score.

   Additional nodes from [GitHub Pages](https://github.com/strimmlarn/ComfyUI-Strimmlarns-Aesthetic-Score/) - These have been modified to improve performance and add an option to store the model in RAM, which significantly reduces generation time:
   - `CalculateAestheticScore`: An optimized version of the original, with an option to keep the model loaded in RAM.
   - `AestheticScoreSorter`: Sorts the images by score.
   - `AestheticModel`: Loads the aesthetic model.
2. **ImageCaptioningNode.py** - Implements nodes for image captioning and classification:
   - `ImageCaptioningNode`: Provides a caption for the image using BLIP model.
      - _Input_: `image` (IMAGE)
      - _Output_: String caption.
   - `FoodCategoryClassifierNode`: Classifies food categories in images.
      - _Input_: `image` (IMAGE)
      - _Output_: Top 5 food categories with probabilities.
   - `AgeClassifierNode`: Classifies the age range in images.
      - _Input_: `image` (IMAGE)
      - _Output_: Top 5 age ranges with probabilities.
   - `ArtOrHumanClassifierNode`: Detects if an image is AI-generated or human-made.
      - _Input_: `image` (IMAGE), `show_on_node` (BOOL)
      - _Output_: Artificial and human probabilities.
   - `DocumentClassificationNode`: Classifies document types.
      - _Input_: `image` (IMAGE)
      - _Output_: Document type index and name.
   - `NSFWClassifierNode`: Classifies content safety levels.
      - _Input_: `image` (IMAGE), `show_on_node` (BOOL), `threshold` (FLOAT)
      - _Output_: 
         - Classification report (STRING)
         - SFW Score (FLOAT)
         - NSFW Score (FLOAT)
         - Is SFW (BOOLEAN)
         - Is NSFW (BOOLEAN)
   - `WatermarkDetectionNode`: Detects watermarks in images using EfficientNet.
      - _Input_: `image` (IMAGE), `show_on_node` (BOOL), `threshold` (FLOAT)
      - _Output_:
         - Classification report (STRING)
         - Clean Score (FLOAT)
         - Watermark Score (FLOAT)
         - Is Clean (BOOLEAN)
         - Has Watermark (BOOLEAN)

3. **SegformerNode.py** - Handles semantic segmentation of images:
   - `SegformerNode`: Performs semantic segmentation with multiple model options.
      - _Input_: `image` (IMAGE), `model_name` (STRING), `normalize_mask` (BOOL), `binary_mask` (BOOL), `resize_mode` (STRING), `invert_mask` (BOOL), `show_preview` (BOOL), `return_individual_masks` (BOOL), `post_process` (STRING), `post_process_radius` (INT), `segment_groups` (STRING)
      - _Output_: Segmented image, mask, info, and preview.
   - `SegformerNodeMasks`: Creates individual segment masks.
      - _Input_: `image` (IMAGE), `segments_to_merge` (STRING), `model_name` (STRING)
      - _Output_: Image, mask, and segment info.
   - `SegformerNodeMergeSegments`: Merges and processes segments with advanced options.
      - _Input_: `image` (IMAGE), `segments_to_merge_str` (STRING), `model_name` (STRING), `normalize_mask` (BOOL), `binary_mask` (BOOL), `resize_mode` (STRING), `invert_mask` (BOOL), `show_preview` (BOOL), `blur_radius` (INT), `dilation_radius` (INT), `intensity` (FLOAT), `ceiling` (FLOAT)
      - _Output_: Processed image, mask, info, and preview.
   - `SeedIncrementerNode`: Manages seed incrementation for workflows.
      - _Input_: `seed` (INT), `IncrementAt` (INT)
      - _Output_: Seed string, seed int, subseed string, subseed int.
   - `StepCfgIncrementNode`: Handles step and configuration increments.
      - _Input_: `seed` (INT), `cfg_start` (INT), `steps_start` (INT), `image_steps` (INT), `max_steps` (INT)
      - _Output_: CFG and steps values.

## Requirements

The project requires the following Python libraries:

- torch
- transformers
- Pillow (PIL)
- matplotlib
- numpy
- scipy
- huggingface_hub
- torchvision

## Installation

1. Install the required Python packages:
```bash
pip install torch transformers pillow matplotlib numpy scipy huggingface_hub torchvision
```

2. Clone this repository into your ComfyUI custom_nodes directory:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/YourUsername/ComfyUI-LexTools.git
```

3. Restart ComfyUI to load the new nodes.

## Usage

The nodes will appear in the ComfyUI interface under the "LexTools" category, organized into subcategories:
- LexTools/ImageProcessing/Segmentation
- LexTools/ImageProcessing/Classification
- LexTools/ImageProcessing/Captioning
- LexTools/Utilities

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

