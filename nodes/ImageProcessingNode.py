import hashlib
from fastapi import FastAPI
import torch, time
import io

import comfy.samplers
from matplotlib import transforms
from PIL import Image, ImageFilter, ImageEnhance, ImageOps, ImageDraw, ImageChops, ImageFont
import numpy as np
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
folder_paths.folder_names_and_paths["aesthetic"] = ([os.path.join(folder_paths.models_dir,"aesthetic")], folder_paths.supported_pt_extensions)


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


class ImageAspectPadNode:

    @classmethod
    def INPUT_TYPES(s):
        global aspect_ratios  # Assuming aspect_ratios is a list of aspect ratio strings
        return {
            "required": {
                "image": ("IMAGE",),
                "aspect_ratio": (aspect_ratios, {"default": aspect_ratios[0]}),
                "invert_ratio": (["true", "false"], {"default": "false"}),
                "feathering": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
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

    def expand_image(self, image, aspect_ratio, invert_ratio, feathering, left_padding, right_padding, top_padding, bottom_padding,show_on_node):
   
        d1, d2, d3, d4 = image.size()
        aspect_ratio = float(aspect_ratio.split('/')[0]) / float(aspect_ratio.split('/')[1])
        if invert_ratio == "true":
            aspect_ratio = 1.0 / aspect_ratio

        image_aspect_ratio = d3 / d2
        if image_aspect_ratio > aspect_ratio:
            pad_height = int(d3 / aspect_ratio) - d2
            top_padding += pad_height // 2
            bottom_padding += pad_height - top_padding
        else:
            pad_width = int(d2 * aspect_ratio) - d3
            left_padding += pad_width // 2
            right_padding += pad_width - left_padding
        new_image = torch.zeros(
            (d1, d2 + top_padding + bottom_padding, d3 + left_padding + right_padding, d4),
            dtype=torch.float32,
        )
        new_image[:, top_padding:top_padding + d2, left_padding:left_padding + d3, :] = image

        mask = torch.ones(
            (d2 + top_padding + bottom_padding, d3 + left_padding + right_padding),
            dtype=torch.float32,
        )

        t = torch.zeros(
            (d2, d3),
            dtype=torch.float32
        )

        if feathering > 0 and feathering * 2 < d2 and feathering * 2 < d3:

            for i in range(d2):
                for j in range(d3):
                    dt = i if top_padding != 0 else d2
                    db = d2 - i if bottom_padding != 0 else d2

                    dl = j if left_padding != 0 else d3
                    dr = d3 - j if right_padding != 0 else d3

                    d = min(dt, db, dl, dr)

                    if d >= feathering:
                        continue

                    v = (feathering - d) / feathering

                    t[i, j] = v * v

        mask[top_padding:top_padding + d2, left_padding:left_padding + d3] = t
   
            
        output_ui = {}
        if show_on_node ==1:
            output_ui = {"ui": {"images": [new_image]}}



        return  (new_image, mask, output_ui)

    


class ImageScaleToMin:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",)},
                "optional":{"MinScalePix": ("FLOAT", {"default": 512, "min": 0.0, "max": 2056, "step": 1}),}}

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "calculate_scale"

    CATEGORY = "LexTools/ImageProcessing/upscaling"

    def calculate_scale(self, image,MinScalePix):
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
                "image": ("IMAGE",),
                "score": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0}),
                "threshold": ("FLOAT", {"default": 5.0, "min": -100.0, "max": 100.0}),
                "show_on_node": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "FLOAT")
    FUNCTION = "filter_image"
    CATEGORY = "LexTools/ImageProcessing/Filtering"

    def filter_image(self, image, score, threshold, show_on_node):
        try:
            if float(score) >= float(threshold):
                score_text = f"Score {score:.2f} >= Threshold {threshold:.2f}\nImage Passed"
                output_ui = {"text": [score_text]} if show_on_node else {}
                return {"result": (image, float(score)), "ui": output_ui}
            else:
                score_text = f"Score {score:.2f} < Threshold {threshold:.2f}\nImage Filtered"
                output_ui = {"text": [score_text]} if show_on_node else {}
                return {"result": (torch.zeros_like(image), float(score)), "ui": output_ui}
        except Exception as e:
            print(f"Error filtering image: {str(e)}")
            return {"result": (image, 0.0), "ui": {"text": [str(e)]} if show_on_node else {}}

class ImageQualityScoreNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "aesthetic_score": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0}),
                "image_score_good": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0}),
                "image_score_bad": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0}),
                "ai_score_artificial": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0}),
                "ai_score_human": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0}),
                "weight_good_score": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0}),
                "weight_aesthetic_score": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0}),
                "weight_bad_score": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0}),
                "weight_AIDetection": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0}),
                "weight_HumanDetection": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0}),
                "MultiplyScoreBy": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0}),
                "show_on_node": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "calculate_score"
    CATEGORY = "LexTools/ImageProcessing/Scoring"

    def calculate_score(self, aesthetic_score, image_score_good, image_score_bad, ai_score_artificial, ai_score_human,
                       weight_good_score, weight_aesthetic_score, weight_bad_score, weight_AIDetection, weight_HumanDetection,
                       MultiplyScoreBy, show_on_node):
        try:
            # Calculate weighted scores
            weighted_aesthetic = float(aesthetic_score) * weight_aesthetic_score
            weighted_good = float(image_score_good) * weight_good_score
            weighted_bad = float(image_score_bad) * weight_bad_score
            weighted_ai = float(ai_score_artificial) * weight_AIDetection
            weighted_human = float(ai_score_human) * weight_HumanDetection

            # Calculate total score
            total_score = (weighted_aesthetic + weighted_good - weighted_bad + weighted_human - weighted_ai) * MultiplyScoreBy

            # Format score for display
            score_text = f"Score: {total_score:.2f}\n"
            score_text += f"Aesthetic (w:{weight_aesthetic_score:.1f}): {aesthetic_score:.2f}\n"
            score_text += f"Good (w:{weight_good_score:.1f}): {image_score_good:.2f}\n"
            score_text += f"Bad (w:{weight_bad_score:.1f}): {image_score_bad:.2f}\n"
            score_text += f"AI (w:{weight_AIDetection:.1f}): {ai_score_artificial:.2f}\n"
            score_text += f"Human (w:{weight_HumanDetection:.1f}): {ai_score_human:.2f}\n"
            score_text += f"Multiplier: {MultiplyScoreBy:.1f}"

            output_ui = {"text": [score_text]} if show_on_node else {}

            return {"result": (float(total_score),), "ui": output_ui}
        except Exception as e:
            print(f"Error calculating score: {str(e)}")
            return {"result": (0.0,), "ui": {"text": [str(e)]} if show_on_node else {}}


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
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            #nn.ReLU(),
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
        x_hat =fastapiself.layers(x)
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
    return { "required": {"model_name": (folder_paths.get_filename_list("aesthetic"), )}}
  RETURN_TYPES = ("AESTHETIC_MODEL",)
  FUNCTION = "load_model"
  CATEGORY = "LexTools/ImageProcessing/aestheticscore"
  def load_model(self, model_name):
    #load model
    m_path = folder_paths.folder_names_and_paths["aesthetic"][0]
    m_path2 = os.path.join(m_path[0],model_name)
    return (m_path2,)


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
                "keep_in_memory": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("SCORE",)
    FUNCTION = "execute"
    CATEGORY = "LexTools/ImageProcessing/aestheticscore"

    def execute(self, image, aesthetic_model, keep_in_memory):
        if not self.model2 or not self.preprocess:
            self.model2, self.preprocess = clip.load("ViT-L/14", device=self.device)  #RN50x64 

        m_path2 = aesthetic_model

        if not self.model:
            self.model = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14
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
        prediction = self.model(torch.from_numpy(im_emb_arr).to(self.device).type(torch.cuda.FloatTensor))
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
          "required":{
            "image": ("IMAGE",),
            "score": ("SCORE",),
            "image2": ("IMAGE",),
            "score2": ("SCORE",),
          }
        }
  RETURN_TYPES = ("IMAGE", "SCORE", "IMAGE", "SCORE",)
  FUNCTION = "execute"
  CATEGORY = "LexTools/ImageProcessing/aestheticscore"
  def execute(self,image,score,image2,score2):
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
        if show_on_node ==1:
           output_ui = {"ui": {"STRING": [score_str]}}

        return  (score_int, score_float, score_str, output_ui)

class SamplerPropertiesNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":{
                    "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.5, "round": 0.01}),
                    "denoise": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 1, "step":0.1, "round": 0.01}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
          
                     }
                }

    RETURN_TYPES = ("STRING","INT","FLOAT","FLOAT","STRING","STRING")
    FUNCTION = "sample"

    CATEGORY = "sampling"

    def sample(self, ckpt_name,  steps, cfg, sampler_name, scheduler,denoise):
        pass
        return (ckpt_name, steps, cfg, sampler_name, scheduler,denoise)
    
NODE_CLASS_MAPPINGS = {

    "ImageFilterByIntScoreNode": ImageFilterByIntScoreNode,
    "ImageFilterByFloatScoreNode": ImageFilterByFloatScoreNode,
    "ImageScaleToMin": ImageScaleToMin,
    "ImageAspectPadNode": ImageAspectPadNode,
    "ImageRankingNode": ImageRankingNode,
    "ImageQualityScoreNode": ImageQualityScoreNode,
    "ScoreConverterNode":ScoreConverterNode,
    "MD5ImageHashNode": MD5ImageHashNode,
    "SamplerPropertiesNode": SamplerPropertiesNode,
    "CalculateAestheticScore": CalculateAestheticScore,
    "LoadAesteticModel":AesteticModel,
    "AesthetlcScoreSorter": AesthetlcScoreSorter,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageFilterByIntScoreNode": "Image Filter (Int Score)",
    "ImageFilterByFloatScoreNode": "Image Filter (Float Score)",
    "ImageScaleToMin": "Image Scale To Min",
    "ImageRankingNode": "Image Ranking For Image Reward",
    "ScoreConverterNode":"Score Converter (Aesthetic Score)",
    "MD5ImageHashNode":"MD5 Image Hash",
    "SamplerPropertiesNode":"Sampler input node",
    "LoadAesteticModel": "LoadAesteticModel",
    "CalculateAestheticScore": "CalculateAestheticScore",
    "AesthetlcScoreSorter": "AesthetlcScoreSorter",
    }