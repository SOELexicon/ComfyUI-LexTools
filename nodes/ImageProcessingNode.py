import hashlib
import fastapi
import fastapi
import torch
import time
import io
import cv2

from transformers import AutoModelForCausalLM, AutoTokenizer
from torchvision import transforms
import comfy.samplers
import matplotlib.transforms as mpl_transforms
from PIL import Image, ImageFilter, ImageEnhance, ImageOps, ImageDraw, ImageChops, ImageFont
import numpy as np
from scipy.ndimage import zoom

import comfy.model_management as model_management
import json
import uuid
import os
from warnings import filterwarnings
import pytorch_lightning as pl
import torch.nn as nn
from os.path import join
import clip
import folder_paths

# create path to aesthetic model.
folder_paths.folder_names_and_paths["aesthetic"] = ([os.path.join(
    folder_paths.models_dir, "aesthetic")], folder_paths.supported_pt_extensions)


aspect_ratios = [
    "1/1",  # square
    "4/3",  # standard monitor
    "3/2",  # 35mm film
    "16/9",  # widescreen monitor
    "21/9"   # ultrawide monitor
]

MAX_RESOLUTION = 10240  # adjust this value as needed

# Tensor to PIL


def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# PIL to Tensor


def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

# PIL Hex


def pil2hex(image):
    return hashlib.sha256(np.array(tensor2pil(image)).astype(np.uint16).tobytes()).hexdigest()

# PIL to Mask


def pil2mask(image):
    image_np = np.array(image.convert("L")).astype(np.float32) / 255.0
    mask = torch.from_numpy(image_np)
    return 1.0 - mask

# Mask to PIL


def mask2pil(mask):
    if mask.ndim > 2:
        mask = mask.squeeze(0)
    mask_np = mask.cpu().numpy().astype('uint8')
    mask_pil = Image.fromarray(mask_np, mode="L")
    return mask_pil


def scale_and_print_mask(mask, target_shape=(11, 11)):
    """
    Scale a given mask to a target shape and print it rounded to 4 decimal places.
    """
    # Calculate scaling factors
    scale_x = target_shape[0] / mask.shape[0]
    scale_y = target_shape[1] / mask.shape[1]

    # Rescale the mask
    scaled_mask = zoom(mask, (scale_x, scale_y))

    # Print the scaled mask, rounded to 4 decimal places
    for row in scaled_mask:
        print(", ".join([f"{x:.2f}" for x in row]))


def apply_feathering(mask, feathering_distance=10):
    # Apply Gaussian blur to the mask
    mask_feathered = mask

    if feathering_distance > 0:
        # Generate kernel
        kernel_size = 2 * feathering_distance + 1
        kernel = cv2.getGaussianKernel(kernel_size, feathering_distance)

        # Convert the mask tensor to a numpy array
        mask_np = mask.numpy()

        # Apply Gaussian blur to the mask
        kernel_2d = np.dot(kernel, kernel.T)
        mask_feathered_np = cv2.filter2D(
            mask_np, -1, kernel_2d, borderType=cv2.BORDER_CONSTANT)

        # Convert the result back to a PyTorch tensor
        mask_feathered = torch.tensor(mask_feathered_np, dtype=torch.float32)
    return mask_feathered


def apply_gradient(mask, transition_points, feathering_distance):
    # Ensure feathering_distance is at least 1
    gradient = torch.linspace(0, 1, max(feathering_distance, 1))
    for x, y in transition_points:
        if x + feathering_distance < mask.shape[0]:
            mask[x: x + feathering_distance, y] = gradient
        if x - feathering_distance >= 0:
            mask[x - feathering_distance: x, y] = gradient[::-1]
    return mask


class ImageAspectPadNode:

    @classmethod
    def INPUT_TYPES(cls):
        global aspect_ratios  # Assuming aspect_ratios is a list of aspect ratio strings
        return {
            "required": {
                "image": ("IMAGE",),
                "aspect_ratio": (aspect_ratios, {"default": aspect_ratios[0]}),
                "invert_ratio": (["true", "false"], {"default": "false"}),
                "edge_feathering_distance": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "feathering": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "exclude_out_of_bounds": (["true", "false"], {"default": "false"}),
                "left_padding": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "right_padding": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "top_padding": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "bottom_padding": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
            },
            "optional": {
                "show_on_node": ("INT", {"default": 0}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "expand_image"
    OUTPUT_NODE = True

    CATEGORY = "LexTools/ImageProcessing/AspectPad"

    def expand_image(self, image, aspect_ratio, invert_ratio, feathering, left_padding, right_padding, top_padding, bottom_padding, show_on_node, exclude_out_of_bounds, edge_feathering_distance):
        debug_info = {}  # Initialize debug info dictionary
        debug = True
        try:
            # Initial setup
            d1, d2, d3, d4 = image.size()
            aspect_ratio = float(aspect_ratio.split(
                '/')[0]) / float(aspect_ratio.split('/')[1])
            if invert_ratio == "true":
                aspect_ratio = 1.0 / aspect_ratio

            # Padding calculations
            image_aspect_ratio = d3 / d2
            if image_aspect_ratio > aspect_ratio:
                pad_height = int(d3 / aspect_ratio) - d2
                top_padding += pad_height // 2
                bottom_padding += pad_height - top_padding
            else:
                pad_width = int(d2 * aspect_ratio) - d3
                left_padding += pad_width // 2
                right_padding += pad_width - left_padding

            # Debug Information
            debug_info['image_size'] = (d1, d2, d3, d4)
            debug_info['padding'] = (
                top_padding, bottom_padding, left_padding, right_padding)

            # Identify the mask boundary
            boundary_top = top_padding
            boundary_bottom = top_padding + d2
            boundary_left = left_padding
            boundary_right = left_padding + d3

            # Initialize new image and mask
            new_image = torch.zeros((d1, d2 + top_padding + bottom_padding,
                                    d3 + left_padding + right_padding, d4), dtype=torch.float32)
            new_image[:, top_padding:top_padding + d2,
                      left_padding:left_padding + d3, :] = image
            mask = torch.ones((d2 + top_padding + bottom_padding,
                              d3 + left_padding + right_padding), dtype=torch.float32)
            mask[top_padding:top_padding + d2,
                 left_padding:left_padding + d3] = 0
            if debug == True:
                scale_and_print_mask(mask)

            # Apply edge feathering to the identified boundary within mask
            transition_points = []
            for i in range(1, mask.shape[0]):
                for j in range(mask.shape[1]):
                    if mask[i, j] != mask[i-1, j]:
                        transition_points.append((i, j))
            for i in range(mask.shape[0]):
                for j in range(1, mask.shape[1]):
                    if mask[i, j] != mask[i, j-1]:
                        transition_points.append((i, j))
            if debug == True:
                print("Transition Points:", transition_points)  # Debug line

            # Check if "exclude_out_of_bounds" is set to "true"
            if exclude_out_of_bounds == "true":
                # Create a mask that excludes the areas touching the bounds
                inner_mask = torch.zeros_like(mask)
                inner_mask[boundary_top:boundary_bottom, boundary_left:boundary_right] = 1
                inner_mask = 1 - inner_mask  # Invert the inner mask
                
                # Apply the inner mask to the original mask
                mask = mask * inner_mask  

                # Filter transition points to only include those within the bounds
                transition_points = [point for point in transition_points if boundary_top <= point[0] < boundary_bottom and boundary_left <= point[1] < boundary_right]

            # Apply edge feathering to the identified boundary within the new mask
            if edge_feathering_distance > 0:
                try:
                    mask = apply_gradient(mask, transition_points, edge_feathering_distance)
                except Exception as e:
                    if debug == True:
                        print("An error occurred:", str(e))
            # Apply edge feathering to the identified boundary within mask
            if debug == True:
                print("Mask After  Before Feather:")  # Debug line
                # Assuming this function prints the mask
                scale_and_print_mask(mask)

            # Apply overall feathering
            if feathering > 0:
                try:
                    mask = apply_feathering(mask, feathering)
                except Exception as e:
                    # scale_and_print_mask(mask)
                    if debug == True:
                        print("An error occurred:", str(e))
            if debug == True:
                print("Mask After  Feather:")  # Debug line

                # Assuming this function prints the mask
                scale_and_print_mask(mask)

            # Debugging output
                print("Debug Information:", debug_info)

            output_ui = {}
            if show_on_node == 1:
                output_ui = {"ui": {"images": [new_image]}}

            return (new_image, mask, output_ui)

        except Exception as e:

            print("An error occurred:", str(e))
            if debug == True:
                print("Debug Information:", debug_info)
            raise  # Re-raise the caught exception for further handling


class ImageRankingNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "score": ("INT",),
                "prompt": ("STRING",),
                "image_path": ("STRING",),
                "json_file_path": ("STRING",)  # JSON file path
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "rank_image"
    CATEGORY = "LexTools/ImageProcessing/Ranking"

    def rank_image(self, score, prompt, image_path, json_file_path):
        # Load JSON data from file
        with open(json_file_path, 'r') as f:
            data = json.load(f)

        # Check if prompt exists in data
        for record in data:
            if record["prompt"] == prompt:
                # Prompt exists, append image path and score
                record["generations"].append(image_path)
                record["ranking"].append(score)
                break
        else:
            # Prompt does not exist, create new record
            new_id = str(uuid.uuid4())  # Generate a unique ID
            new_record = {
                "id": new_id,
                "prompt": prompt,
                "generations": [image_path],
                "ranking": [score]
            }
            data.append(new_record)

        # Save updated data back to JSON file
        with open(json_file_path, 'w') as f:
            json.dump(data, f)


class AutoModelForCausalLMNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "MESSAGE": ("STRING",),
            "MaxTokens": ("INTEGER",)
        }}

    RETURN_TYPES = ("STRING")
    FUNCTION = "caption"

    CATEGORY = "LexTools/TextGeneration"

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.1")
        self.model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.1")

    def caption(self, MESSAGE, MaxTokens):

        device = "cuda"  # the device to load the model onto
        messages = [
            {"role": "user", "content": MESSAGE},

        ]
        encodeds = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt")

        model_inputs = encodeds.to(device)
        self.model.to(device)

        generated_ids = self.model.generate(
            model_inputs, max_new_tokens=MaxTokens, do_sample=True)
        decoded = self.tokenizer.batch_decode(generated_ids)

        return (decoded[0])


class ImageScaleToMin:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",)},
                "optional": {"MinScalePix": ("FLOAT", {"default": 512, "min": 0.0, "max": 2056, "step": 1}), }}

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "calculate_scale"

    CATEGORY = "LexTools/ImageProcessing/upscaling"

    def calculate_scale(self, image, MinScalePix):
        d1, height, width, d4 = image.shape
        min_dim = min(width, height)
        scale = MinScalePix / min_dim
        return (scale,)


class ImageFilterByIntScoreNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "score": ("INT", {"default": 0}),
                "threshold": ("INT", {"default": 0}),
                "image": ("IMAGE", {"default": None}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "filter_image_by_score"
    CATEGORY = "LexTools/ImageProcessing/Scores"

    def filter_image_by_score(self, score, threshold, image):
        # If score > threshold, return the image, otherwise return None
        if score < threshold:
            pass
        else:
            return (image,)


class ImageFilterByFloatScoreNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "score": ("FLOAT", {"default": 0.0}),
                "threshold": ("FLOAT", {"default": 0.0}),
                "image": ("IMAGE", {"default": None}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "filter_image_by_score"
    CATEGORY = "LexTools/ImageProcessing/Scores"

    def filter_image_by_score(self, score, threshold, image):
        # If score > threshold, return the image, otherwise return None
        if score < threshold:
            pass
        else:
            return (image,)


class ImageQualityScoreNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "aesthetic_score": ("INT", {"default": None}),
                "ai_score_artificial": ("FLOAT", {"default": None}),
                "ai_score_human": ("FLOAT", {"default": None}),
                "show_on_node": ("INT", {"default": 0}),
            },
            "optional": {
                "image_score_good": ("FLOAT", {"default": 0}),
                "image_score_bad": ("FLOAT", {"default": 0}),
                "weight_good_score": ("FLOAT", {"default": 1}),
                "weight_aesthetic_score": ("FLOAT", {"default": 1.0}),
                "weight_bad_score": ("FLOAT", {"default": 1.0}),
                "weight_AIDetection": ("FLOAT", {"default": 1.0}),
                "weight_HumanDetection": ("FLOAT", {"default": 1.0}),
                "MultiplyScoreBy": ("FLOAT", {"default": 100000}),
            },
        }
    OUTPUT_NODE = True

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "calculate_score"
    CATEGORY = "LexTools/ImageProcessing/Scores"

    def calculate_score(self, image_score_good, image_score_bad, aesthetic_score, ai_score_artificial, ai_score_human, weight_good_score, weight_aesthetic_score, weight_bad_score, weight_AIDetection, MultiplyScoreBy, show_on_node, weight_HumanDetection):
        # Define the weights and maximum possible values
        maxA, maxB, maxC = 3, 3, 1000
        # Compute the exponential effect of the AI score
        ai_score_artificial_exp = 10 ** ai_score_artificial
        # Compute the final score according to the provided formula
        final_score = ((((((image_score_good + maxA) / (2 * maxA) * weight_good_score) + (aesthetic_score / maxC) * weight_bad_score) / (weight_good_score + weight_bad_score)) -
                       weight_aesthetic_score * ((image_score_bad + maxB) / (2 * maxB))) * ((weight_HumanDetection * (ai_score_human))-(weight_AIDetection * (ai_score_artificial_exp)))) * MultiplyScoreBy

        # Prepare the output UI
        return (final_score, {"ui": {"STRING": [final_score]}})


#
# Class taken from https://github.com/christophschuhmann/improved-aesthetic-predictor simple_inference.py
#
class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            # nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            # nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = fastapiself.layers(x)
        loss = fastapi.mse_loss(x_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def normalized(a, axis=-1, order=2):
        import numpy as np  # pylint: disable=import-outside-toplevel
        l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
        l2[l2 == 0] = 1
        return a / np.expand_dims(l2, axis)


class AesteticModel:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model_name": (folder_paths.get_filename_list("aesthetic"), )}}
    RETURN_TYPES = ("AESTHETIC_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "LexTools/ImageProcessing/aestheticscore"

    def load_model(self, model_name):
        # load model
        m_path = folder_paths.folder_names_and_paths["aesthetic"][0]
        m_path2 = os.path.join(m_path[0], model_name)
        return (m_path2,)


class SaturationMatchingNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "working_image": ("IMAGE",),
                "master_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "LexTools/ImageProcessing/SaturationMatching"

    def __init__(self):
        self.working_image = None
        self.master_image = None

    def calculate_saturation(self, tensor_image):

        if len(tensor_image.shape) == 4:
            # Take the first image from the batch
            tensor_image = tensor_image[0]

        # Convert tensor to numpy array and then to PIL Image
        img = (tensor_image * 255).to(torch.uint8).numpy()
        pil_image = Image.fromarray(img, mode='RGB')

        # Convert to HSV
        hsv_image = pil_image.convert('HSV')
        s_channel = np.array(hsv_image)[:, :, 1]

        # Calculate average saturation
        avg_saturation = np.mean(s_channel)
        return avg_saturation

    def adjust_saturation(self, tensor_image, target_saturation):

        original_shape = tensor_image.shape
        if len(tensor_image.shape) == 4:
            # Take the first image from the batch
            tensor_image = tensor_image[0]

        # Convert tensor to numpy array and then to PIL Image
        img = (tensor_image * 255).to(torch.uint8).cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        pil_image = Image.fromarray(img, mode='RGB')

        # Convert to HSV
        hsv_image = pil_image.convert('HSV')
        hsv_array = np.array(hsv_image)

        # Calculate current average saturation
        current_saturation = np.mean(hsv_array[:, :, 1])

        # Calculate adjustment factor
        factor = target_saturation / current_saturation

        # Adjust saturation
        hsv_array[:, :, 1] = np.clip(
            hsv_array[:, :, 1] * factor, 0, 255).astype(np.uint8)

        # Convert back to PIL Image and then to tensor
        adjusted_hsv_image = Image.fromarray(hsv_array, 'HSV')
        adjusted_rgb_image = adjusted_hsv_image.convert('RGB')
        tensor_transform = transforms.ToTensor()
        adjusted_tensor = tensor_transform(adjusted_rgb_image)

        # Reshape to match the original tensor shape
        if len(original_shape) == 4:
            adjusted_tensor = adjusted_tensor.unsqueeze(0)

        return adjusted_tensor

    def run(self, working_image, master_image):
        self.working_image = working_image
        self.master_image = master_image

        # Calculate target saturation from master image
        target_saturation = self.calculate_saturation(self.master_image)

        # Adjust the saturation of the working image
        adjusted_image = self.adjust_saturation(
            self.working_image, target_saturation)

        return adjusted_image


class CalculateAestheticScore:
    device = "cuda"
    model2 = None
    preprocess = None
    model = None

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "aesthetic_model": ("AESTHETIC_MODEL",),
            },
            "optional": {
                "keep_in_memory": ("BOOL", {"default": True}),
            }
        }

    RETURN_TYPES = ("SCORE",)
    FUNCTION = "execute"
    CATEGORY = "LexTools/ImageProcessing/aestheticscore"

    def execute(self, image, aesthetic_model, keep_in_memory):
        if not self.model2 or not self.preprocess:
            self.model2, self.preprocess = clip.load(
                "ViT-L/14", device=self.device)  # RN50x64

        m_path2 = aesthetic_model

        if not self.model:
            # CLIP embedding dim is 768 for CLIP ViT L 14
            self.model = MLP(768)
            s = torch.load(m_path2)
            self.model.load_state_dict(s)
            self.model.to(self.device)

        self.model.eval()

        tensor_image = image[0]
        img = (tensor_image * 255).to(torch.uint8).numpy()
        pil_image = Image.fromarray(img, mode='RGB')

        # Use the class variable preprocess
        image2 = self.preprocess(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Use the class variable model2
            image_features = self.model2.encode_image(image2)

        im_emb_arr = normalized(image_features.cpu().detach().numpy())
        prediction = self.model(torch.from_numpy(im_emb_arr).to(
            self.device).type(torch.cuda.FloatTensor))
        final_prediction = int(float(prediction[0])*100)

        if not keep_in_memory:
            self.model = None
            self.model2 = None
            self.preprocess = None

        return (final_prediction,)


class MD5ImageHashNode:
    device = "cuda"

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    CATEGORY = "LexTools/ImageProcessing/md5hash"

    def execute(self, image):
        tensor_image = image[0]

        # Convert the tensor to a PIL image
        img = (tensor_image * 255).to(torch.uint8).cpu().numpy()
        pil_image = Image.fromarray(img, mode='RGB')

        # Convert PIL image to bytes
        image_byte_arr = io.BytesIO()
        pil_image.save(image_byte_arr, format='PNG')
        image_byte_arr = image_byte_arr.getvalue()

        # Calculate MD5 hash
        m = hashlib.md5()
        m.update(image_byte_arr)
        md5_hash = m.hexdigest()

        return (md5_hash,)


class AesthetlcScoreSorter:
    def __init__(self):
        pass
    pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "score": ("SCORE",),
                "image2": ("IMAGE",),
                "score2": ("SCORE",),
            }
        }
    RETURN_TYPES = ("IMAGE", "SCORE", "IMAGE", "SCORE",)
    FUNCTION = "execute"
    CATEGORY = "LexTools/ImageProcessing/aestheticscore"

    def execute(self, image, score, image2, score2):
        if score >= score2:
            return (image, score, image2, score2,)
        else:
            return (image2, score2, image, score,)


class ScoreConverterNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "score": ("SCORE", {"default": 0.0}),
            },
            "optional": {
                "show_on_node": ("INT", {"default": 0}),
            },
        }

    RETURN_TYPES = ("INT", "FLOAT", "STRING",)
    FUNCTION = "convert_score"
    CATEGORY = "LexTools/ImageProcessing/Scores"
    OUTPUT_NODE = True

    def convert_score(self, score, show_on_node):
        # Convert the score to an integer, float, and string
        score_int = int(score)
        score_float = float(score)
        score_str = str(score)

        # Prepare the output UI
        output_ui = {}
        if show_on_node == 1:
            output_ui = {"ui": {"STRING": [score_str]}}

        return (score_int, score_float, score_str, output_ui)


class SamplerPropertiesNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
            "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
            "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.5, "round": 0.01}),
            "denoise": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 1, "step": 0.1, "round": 0.01}),
            "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
            "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),

        }
        }

    RETURN_TYPES = ("STRING", "INT", "FLOAT", "FLOAT", "STRING", "STRING")
    FUNCTION = "sample"

    CATEGORY = "sampling"

    def sample(self, ckpt_name,  steps, cfg, sampler_name, scheduler, denoise):
        pass
        return (ckpt_name, steps, cfg, sampler_name, scheduler, denoise)


NODE_CLASS_MAPPINGS = {

    "ImageFilterByIntScoreNode": ImageFilterByIntScoreNode,
    "ImageFilterByFloatScoreNode": ImageFilterByFloatScoreNode,
    "ImageScaleToMin": ImageScaleToMin,
    "ImageAspectPadNode": ImageAspectPadNode,
    "ImageRankingNode": ImageRankingNode,
    "ImageQualityScoreNode": ImageQualityScoreNode,
    "ScoreConverterNode": ScoreConverterNode,
    "MD5ImageHashNode": MD5ImageHashNode,
    "SamplerPropertiesNode": SamplerPropertiesNode,

    "SaturationMatchingNode": SaturationMatchingNode
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageFilterByIntScoreNode": "Image Filter (Int Score)",
    "ImageFilterByFloatScoreNode": "Image Filter (Float Score)",
    "ImageScaleToMin": "Image Scale To Min",
    "ImageRankingNode": "Image Ranking For Image Reward",
    "ScoreConverterNode": "Score Converter (Aesthetic Score)",
    "MD5ImageHashNode": "MD5 Image Hash",
    "SamplerPropertiesNode": "Property Output Node.",
    "SaturationMatchingNode": "SaturationMatchingNode",

}
