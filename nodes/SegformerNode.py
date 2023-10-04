import torch
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image,ImageOps,ImageFilter
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import io
from scipy.ndimage import binary_dilation


model_names = [
                "enes361/segformer_b2_clothes",
                "mattmdjaga/segformer_b0_clothes",
                "mattmdjaga/segformer_b2_clothes",
                "DiTo97/binarization-segformer-b3",
                "s3nh/SegFormer-b0-person-segmentation",
                "venture361/clothes_segmentation", 
                "matei-dorian/segformer-b5-finetuned-human-parsing",
                "Lexic0n/segformer-b0-finetuned-human-parsing",
                "sam1120/segformer-b0-finetuned-neurosymbolic-contingency-bag1-v0.1-v0",
                "ehsanhallo/segformer-b0-scene-parse-150"
                ]

class SegformerNode:
    @classmethod
    def INPUT_TYPES(cls):
        global model_names  # Assuming model_names is a list of model names
        return {
            "required": {
                "image": ("IMAGE", {"default": None}),
                "model_name": (model_names, {"default": model_names[0]}),
            

            },
        }

    RETURN_TYPES = ("IMAGE","MASK", "STRING")
    FUNCTION = "segment_image"
    CATEGORY = "LexTools/ImageProcessing/Segmentation"

    def __init__(self):
        pass
        # self.processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
        # self.model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
        
    def segment_image(self, image,model_name,):
        show_on_node = False
        self.processor = SegformerImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForSemanticSegmentation.from_pretrained(model_name)
        i = 255. * image[0].cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        inputs   = self.processor(images=img, return_tensors="pt")

        outputs = self.model(**inputs)
        logits = outputs.logits.cpu()

        upsampled_logits = nn.functional.interpolate(
            logits,
            size=img.size[::-1],
            mode="bilinear",
            align_corners=False,
        )

        pred_seg = upsampled_logits.argmax(dim=1)[0]

        # Convert the matplotlib figure to a PIL Image and return it
        fig = plt.figure()
        plt.imshow(pred_seg)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img2 = Image.open(buf)
        
        i = ImageOps.exif_transpose(img2)
        if i.getbands() != ("R", "G", "B", "A"):
            i = i.convert("RGBA")

            
        img2 = np.array(img2).astype(np.float32) / 255.0
        img2 = torch.from_numpy(img2)[None,]

        if 'A' in i.getbands():
            mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
        # Get the unique segments in the image
        unique_segments = np.unique(pred_seg)

        # Create a string with the information for each segment
        segment_info = []
        for segment in unique_segments:
            # Get the name of the segment from the model's configuration
            segment_name = self.model.config.id2label[segment]

            # Here, you would replace these values with the actual accuracy and IoU for the segment


            segment_info.append(f"Segment {segment}: {segment_name}")

        # Join the segment info strings into a single string
        segment_info_str = "\n".join(segment_info)


        output_ui =  {"images": [img2]} if show_on_node else {}

        return {"result": (img2,mask, segment_info_str), "ui": output_ui}

class SegformerNodeMasks:
    @classmethod
    def INPUT_TYPES(cls):
        global model_names  # Assuming model_names is a list of model names
        return {
            "required": {
                "image": ("IMAGE", {"default": None}),
                "segments_to_merge": ("STRING", {"default": "0"}),
                "model_name": (model_names, {"default": model_names[0]})
              
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    FUNCTION = "segment_image"
    CATEGORY = "LexTools/ImageProcessing/Segmentation"

    def __init__(self):
        pass

    # Function to segment the image and return the merged segments as per the provided indices
    def segment_image(self, image, segments_to_merge, model_name):
        # Convert the segments_to_merge from string to list of integers
        show_on_node=False
        segments_to_merge = list(map(int, segments_to_merge.split(',')))

        # Load the pretrained models and processors
        self.processor = SegformerImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForSemanticSegmentation.from_pretrained(model_name)

        # Preprocess the image
        i = 255. * image[0].cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        inputs = self.processor(images=img, return_tensors="pt")

        # Get the outputs from the model
        outputs = self.model(**inputs)
        logits = outputs.logits.cpu()

        # Upsample the logits to match the original image size
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=img.size[::-1],
            mode="bilinear",
            align_corners=False,
        )

        # Get the predicted segments
        pred_seg = upsampled_logits.argmax(dim=1)[0]
        unique_segments = np.unique(pred_seg)

        # Initialize lists to hold the segmented images and masks
        segmented_images = []
        masks = []
        
        # Iterate over the unique segments
        for segment in unique_segments:
            # Create a binary mask for the current segment
            mask = np.where(pred_seg == segment, 1, 0).astype(np.uint8)
            # Upsample the mask to match the original image size
            mask = nn.functional.interpolate(torch.from_numpy(mask)[None, None,], size=img.size[::-1], mode="nearest")[0]
            
            # Apply the mask to the original image to get the segmented image
            segmented_image = img * mask.numpy()[0, ..., None]
            segmented_image_pil = Image.fromarray(segmented_image)

            # Convert the segmented image and mask to tensors
            img2 = torch.from_numpy(np.array(segmented_image_pil).astype(np.float32) / 255.0)[None,]
            segmented_images.append(img2)
            masks.append(mask)

        # Initialize a mask of zeros with the same size as the other masks
        merged_mask = torch.zeros_like(masks[0])

        # Iterate over the segments to merge
        for segment in segments_to_merge:
            # Check if the segment index is valid
            if segment < len(masks):
                # Add the current mask to the merged mask
                merged_mask += masks[segment]
            else:
                # Raise an error if the segment index is invalid
                raise ValueError(f"Segment {segment} is out of range. There are only {len(masks)} segments.")

        # Get the merged image by applying the merged mask to the original image
        merged_image = img * merged_mask.numpy()[0, ..., None]
        merged_image_pil = Image.fromarray(merged_image)
        img2 = torch.from_numpy(np.array(merged_image_pil).astype(np.float32) / 255.0)[None,]


        output_ui =  {"images": [img2]} if show_on_node else {}

        return {"result": (img2, merged_mask, 'Merged Segments'), "ui": output_ui}
    


class SegformerNodeMergeSegments:
    @classmethod
    def INPUT_TYPES(cls):
        global model_names  
        return {
            "required": {
                "image": ("IMAGE", {"default": None}),
                "segments_to_merge_str": ("STRING", {"default": ""}),
                "model_name": (model_names, {"default": model_names[0]}),
                "blur_radius": ("INT", {"default": 0}),
                "dilation_radius": ("INT", {"default": 0}),  # Added dilation_radius
                "intensity": ("FLOAT", {"default": 1.0}),  # Added intensity
                "ceiling": ("FLOAT", {"default": 1.0}),  # Added ceiling

            },
        }

    OUTPUT_NODE = True

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    FUNCTION = "merge_segments"
    CATEGORY = "LexTools/ImageProcessing/Segmentation"

    def __init__(self):
        pass

    def merge_segments(self, image, segments_to_merge_str, model_name, blur_radius, dilation_radius, intensity, ceiling):  # Added dilation_radius in the arguments
       
        show_on_node=False
    
        try:
            self.processor = SegformerImageProcessor.from_pretrained(model_name)
        except Exception:
            print(f"Failed to load preprocessor for model {model_name}. Using preprocessor from mattmdjaga/segformer_b2_clothes instead.")
            self.processor = SegformerImageProcessor.from_pretrained("matei-dorian/segformer-b5-finetuned-human-parsing")
        self.model = AutoModelForSemanticSegmentation.from_pretrained(model_name)
            
        i = 255. * image[0].cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        inputs = self.processor(images=img, return_tensors="pt")

        outputs = self.model(**inputs)
        logits = outputs.logits.cpu()

        upsampled_logits = nn.functional.interpolate(
            logits,
            size=img.size[::-1],
            mode="bilinear",
            align_corners=False,
        )

        pred_seg = upsampled_logits.argmax(dim=1)[0].numpy() 
        unique_segments = np.unique(pred_seg)

        segments_to_merge = list(map(int, segments_to_merge_str.split(',')))

        merged_mask = np.zeros_like(pred_seg)

        merged_segments = []
        for segment in unique_segments:
            if segment in segments_to_merge:
                mask = np.where(pred_seg == segment, 1, 0)
                mask = nn.functional.interpolate(torch.from_numpy(mask.astype(np.float32))[None, None,], size=(img.height, img.width), mode="nearest")[0,0].numpy() 

                merged_mask = np.maximum(merged_mask, mask)
                merged_segments.append(segment)

        merged_mask = np.clip(merged_mask * intensity, 0, ceiling)  # Apply intensity and ceiling to the mask
        if dilation_radius > 0:  # Dilate the mask if dilation_radius > 0
            struct = np.ones((2 * dilation_radius + 1, 2 * dilation_radius + 1))
            merged_mask = binary_dilation(merged_mask, structure=struct)
        merged_mask_rgb = np.repeat(merged_mask[..., None], 3, axis=2)  
        if blur_radius > 0:  # Blur the mask if radius > 0
            merged_mask_rgb = Image.fromarray((merged_mask_rgb * 255).astype('uint8'))
            merged_mask_rgb = merged_mask_rgb.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            merged_mask_rgb = np.array(merged_mask_rgb) / 255.0

        merged_image = np.array(img) * merged_mask_rgb

        merged_image_pil = Image.fromarray(merged_image.astype('uint8'))
        if blur_radius > 0:  # Apply blur if radius > 0
            merged_image_pil = merged_image_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            
        img2 = np.array(merged_image_pil).astype(np.float32) / 255.0
        img2 = torch.from_numpy(img2).double()[None,]  

        merged_mask_torch = torch.from_numpy(merged_mask).float()[None,]  # change from double to float

        merged_segments_str = ','.join(map(str, merged_segments))

        output_ui =  {"images": [img2]} if show_on_node else {}

        return {"result": (img2, merged_mask_torch, merged_segments_str), "ui": output_ui}

    
    
class SeedIncrementerNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0,"max": 0xffffffffffffffff}),
                "IncrementAt": ("INT", {"default": 10, "min": 1,"max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = ("STRING", "INT","STRING", "INT")
    FUNCTION = "increment_seed"
    CATEGORY = "LexTools/Utilities"

    def increment_seed(self, seed, IncrementAt):
        # Compute subseed
        subseed = seed // IncrementAt + 1


        # Return seed as string, seed as int, and subseed
        return str(seed), seed, str(subseed),subseed


class StepCfgIncrementNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {"default": 0,"max": 0xffffffffffffffff}),
                "cfg_start": ("INT", {"default": 7}),
                "steps_start": ("INT", {"default": 10}),
                "image_steps": ("INT", {"default": 100}),
                "max_steps": ("INT", {"default": 12}),
            },
        }

    RETURN_TYPES = ("INT", "INT")
    FUNCTION = "calculate_steps_cfg"
    CATEGORY = "LexTools/ImageProcessing/Increment"

    def calculate_steps_cfg(self, seed, cfg_start, steps_start, image_steps, max_steps):
        # Calculate the number of complete cycles
        cycle_count = seed // (image_steps * (max_steps - steps_start + 1))
        
        # Calculate the number of steps within the current cycle
        step_in_cycle = (seed // image_steps) % (max_steps - steps_start + 1)

        # Update cfg and steps values
        cfg = cfg_start + cycle_count
        steps = steps_start + step_in_cycle

        return cfg, steps



    
NODE_CLASS_MAPPINGS = {
    "SegformerNode": SegformerNode,
    "SegformerNodeMasks": SegformerNodeMasks,
    "SegformerNodeMergeSegments": SegformerNodeMergeSegments,
    "SeedIncrementerNode": SeedIncrementerNode,
    "StepCfgIncrementNode": StepCfgIncrementNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SegformerNode": "Segformer Node",
    "SegformerNodeMasks": "Segformer Node Masks",
    "SegformerNodeMergeSegments": "Segformer Node Merge Segments",
    "SeedIncrementerNode": "Seed Incrementer Node",
    "StepCfgIncrementNode": "Step Cfg Increment Node",
}