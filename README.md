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
   - `CalculateAestheticScore`: An optimized version of the original, with an option to keep the model loaded in RAM. (No specific input or output detailed in the provided code)
   - `AesthetlcScoreSorter`: Sorts the images by score. (No specific input or output detailed in the provided code)
   - `AesteticModel`: Loads the aesthetic model. (No specific input or output detailed in the provided code)

2. **ImageCaptioningNode.py** - Implements nodes for image captioning and classification:
   - `ImageCaptioningNode`: Provides a caption for the image.
      - _Input_: `image` (IMAGE)
      - _Output_: String caption.
   - `FoodCategoryNode`: Classifies the food category of an image.
      - _Input_: `image` (IMAGE)
      - _Output_: String category.
   - `AgeClassifierNode`: Classifies the age of a person in the image.
      - _Input_: `image` (IMAGE)
      - _Output_: String age range.
   - `ImageClassifierNode`: General image classification.
      - _Input_: `image` (IMAGE), `show_on_node` (BOOL)
      - _Output_: String label, `artificial_prob` (INT), `human_prob` (INT)
   - `ClassifierNode`: A generic classifier node.
      - _Input_: `image` (IMAGE)
      - _Output_: String label.

3. **SegformerNode.py** - Handles semantic segmentation of images. It includes various nodes such as:
   - `SegformerNode`: Performs segmentation of the image.
      - _Input_: `image` (IMAGE), `model_name` (STRING), `show_on_node` (BOOL)
      - _Output_: Segmented image.
   - `SegformerNodeMasks`: Provides masks for the segmented images.
      - _Input_: No specific input detailed in the provided code.
      - _Output_: Image masks.
   - `SegformerNodeMergeSegments`: Merges certain segments in the segmented image.
      - _Input_: `image` (IMAGE), `segments_to_merge` (STRING), `model_name` (STRING), `blur_radius` (INT), `dilation_radius` (INT), `intensity` (INT), `ceiling` (INT), `show_on_node` (BOOL)
      - _Output_: Image with merged segments.
   - `SeedIncrementerNode`: Increment the seed used for random processes.
      - _Input_: `seed` (INT), `increment_at` (INT)
      - _Output_: Incremented seed.
   - `StepCfgIncrementNode`: Calculates the step configuration for the process.
      - _Input_: `seed` (INT), `cfg_start` (INT), `steps_start` (INT), `img_steps` (INT), `max_steps` (INT)
      - _Output_: Calculated step configuration.

## Requirements

The project primarily uses the following libraries:

- Python
- Torch
- Transformers
- PIL
- Matplotlib
- Numpy
- IO
- Scipy

## Installation

To install the necessary libraries, run:

```bash
pip install torch transformers pillow matplotlib numpy scipy
```

## Contributing
Contributions to this project are welcome. If you find a bug or think of a feature that would benefit the project, please open an issue. If you'd like to contribute code, please open a pull request.

