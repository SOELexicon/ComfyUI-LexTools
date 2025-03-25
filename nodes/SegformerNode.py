import torch
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image,ImageOps,ImageFilter
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import io
from scipy.ndimage import binary_dilation


model_names = [
                "sayeed99/segformer_b3_clothes",
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
        global model_names
        return {
            "required": {
                "image": ("IMAGE", {"default": None}),
                "model_name": (model_names, {"default": model_names[0]}),
                "normalize_mask": ("BOOLEAN", {"default": True}),
                "binary_mask": ("BOOLEAN", {"default": False}),
                "resize_mode": (["nearest", "bilinear", "bicubic"], {"default": "bilinear"}),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "show_preview": ("BOOLEAN", {"default": True}),
                "return_individual_masks": ("BOOLEAN", {"default": False}),
                "post_process": (["none", "erode", "dilate", "smooth"], {"default": "none"}),
                "post_process_radius": ("INT", {"default": 3, "min": 1, "max": 10}),
                "segment_groups": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "IMAGE")  # Added IMAGE for preview
    FUNCTION = "segment_image"
    CATEGORY = "LexTools/ImageProcessing/Segmentation"

    def __init__(self):
        pass
        # self.processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
        # self.model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
        
    def process_mask(self, mask, normalize=True, binary=False, invert=False, post_process="none", radius=3):
        # Convert to float32 if not already
        mask = mask.float()
        
        # Normalize to 0-1 range if requested
        if normalize:
            mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
        
        # Convert to binary if requested
        if binary:
            mask = (mask > 0.5).float()
        
        # Apply post-processing
        if post_process != "none":
            kernel = torch.ones(2 * radius + 1, 2 * radius + 1)
            if post_process == "erode":
                mask = torch.nn.functional.conv2d(
                    mask.unsqueeze(0).unsqueeze(0),
                    kernel.unsqueeze(0).unsqueeze(0),
                    padding=radius
                ).squeeze() < kernel.sum()
            elif post_process == "dilate":
                mask = torch.nn.functional.conv2d(
                    mask.unsqueeze(0).unsqueeze(0),
                    kernel.unsqueeze(0).unsqueeze(0),
                    padding=radius
                ).squeeze() > 0
            elif post_process == "smooth":
                mask = torch.nn.functional.conv2d(
                    mask.unsqueeze(0).unsqueeze(0),
                    kernel.unsqueeze(0).unsqueeze(0),
                    padding=radius
                ).squeeze() / kernel.sum()
            mask = mask.float()
        
        # Invert if requested
        if invert:
            mask = 1 - mask
            
        return mask

    def create_preview(self, image, mask):
        # Create an RGBA preview with the mask as alpha channel
        preview = image.clone()
        preview = torch.cat([preview, mask.unsqueeze(0)], dim=0)
        return preview

    def parse_segment_groups(self, groups_str):
        if not groups_str.strip():
            return {}
        
        groups = {}
        for line in groups_str.split('\n'):
            if ':' in line:
                name, indices = line.split(':')
                indices = [int(i.strip()) for i in indices.split(',') if i.strip()]
                groups[name.strip()] = indices
        return groups

    def segment_image(self, image, model_name, normalize_mask=True, binary_mask=False, 
                     resize_mode="bilinear", invert_mask=False, show_preview=True,
                     return_individual_masks=False, post_process="none", 
                     post_process_radius=3, segment_groups=""):
        show_on_node = False
        self.processor = SegformerImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForSemanticSegmentation.from_pretrained(model_name)
        
        # Process input image
        i = 255. * image[0].cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        inputs = self.processor(images=img, return_tensors="pt")

        # Get model outputs
        outputs = self.model(**inputs)
        logits = outputs.logits.cpu()

        # Upsample logits with specified resize mode
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=img.size[::-1],
            mode=resize_mode,
            align_corners=False if resize_mode != "nearest" else None,
        )

        pred_seg = upsampled_logits.argmax(dim=1)[0]
        
        # Parse segment groups if provided
        segment_groups_dict = self.parse_segment_groups(segment_groups)
        
        # Create individual masks if requested
        individual_masks = {}
        segment_info = []
        
        # Get unique segments and process each
        unique_segments = np.unique(pred_seg.numpy())
        for segment in unique_segments:
            segment_name = self.model.config.id2label[segment]
            segment_info.append(f"Segment {segment}: {segment_name}")
            
            if return_individual_masks:
                mask = (pred_seg == segment).float()
                mask = self.process_mask(mask, normalize_mask, binary_mask, 
                                      invert_mask, post_process, post_process_radius)
                individual_masks[segment_name] = mask

        # Create merged mask based on segment groups
        if segment_groups_dict:
            merged_mask = torch.zeros_like(pred_seg, dtype=torch.float32)
            for group_name, indices in segment_groups_dict.items():
                group_mask = torch.zeros_like(pred_seg, dtype=torch.float32)
                for idx in indices:
                    group_mask = torch.maximum(group_mask, (pred_seg == idx).float())
                merged_mask = torch.maximum(merged_mask, group_mask)
                segment_info.append(f"Group {group_name}: {indices}")
        else:
            merged_mask = torch.ones_like(pred_seg, dtype=torch.float32)

        # Process the final mask
        merged_mask = self.process_mask(merged_mask, normalize_mask, binary_mask, 
                                      invert_mask, post_process, post_process_radius)

        # Create visualization
        fig = plt.figure()
        plt.imshow(pred_seg)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img2 = Image.open(buf)
        
        # Convert visualization to tensor
        i = ImageOps.exif_transpose(img2)
        if i.getbands() != ("R", "G", "B", "A"):
            i = i.convert("RGBA")
        img2 = np.array(img2).astype(np.float32) / 255.0
        img2 = torch.from_numpy(img2)[None,]

        # Create preview if requested
        preview = self.create_preview(image[0], merged_mask) if show_preview else None

        # Join segment info
        segment_info_str = "\n".join(segment_info)
        if return_individual_masks:
            segment_info_str += "\n\nIndividual masks available for: " + ", ".join(individual_masks.keys())

        output_ui = {"images": [img2]} if show_on_node else {}

        # Return results
        return {"result": (img2, merged_mask, segment_info_str, preview if preview is not None else img2), 
                "ui": output_ui}

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

        # Convert the merged mask to byte format (0-255) and ensure correct dimensionality
        merged_mask = (merged_mask > 0).float()  # Convert to binary mask first
        merged_mask = torch.clamp(merged_mask, 0, 1)

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
                "normalize_mask": ("BOOLEAN", {"default": True}),
                "binary_mask": ("BOOLEAN", {"default": False}),
                "resize_mode": (["nearest", "bilinear", "bicubic"], {"default": "bilinear"}),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "show_preview": ("BOOLEAN", {"default": True}),
                "blur_radius": ("INT", {"default": 5, "min": 0, "max": 100}),
                "dilation_radius": ("INT", {"default": 5, "min": 0, "max": 100}),
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
                "ceiling": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
            },
        }

    OUTPUT_NODE = True
    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "IMAGE")  # Added IMAGE for preview
    FUNCTION = "merge_segments"
    CATEGORY = "LexTools/ImageProcessing/Segmentation"

    def __init__(self):
        pass

    def process_mask(self, mask, normalize=True, binary=False, invert=False, blur_radius=0, dilation_radius=0, intensity=1.0, ceiling=1.0):
        # Convert to float32 if not already
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        mask = mask.float()
        
        # Normalize to 0-1 range if requested
        if normalize:
            min_val = mask.min()
            max_val = mask.max()
            if max_val > min_val:
                mask = (mask - min_val) / (max_val - min_val)
        
        # Convert to binary if requested
        if binary:
            mask = (mask > 0.5).float()
        
        # Apply dilation if specified
        if dilation_radius > 0:
            kernel = torch.ones(2 * dilation_radius + 1, 2 * dilation_radius + 1)
            mask = torch.nn.functional.conv2d(
                mask.unsqueeze(0).unsqueeze(0),
                kernel.unsqueeze(0).unsqueeze(0),
                padding=dilation_radius
            ).squeeze() > 0
            mask = mask.float()

        # Apply Gaussian blur for feathering
        if blur_radius > 0:
            mask_np = (mask.numpy() * 255).astype('uint8')
            mask_pil = Image.fromarray(mask_np)
            mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            mask = torch.from_numpy(np.array(mask_pil).astype(np.float32) / 255.0)

        # Apply intensity and ceiling
        mask = torch.clamp(mask * intensity, 0, ceiling)
        
        # Invert if requested
        if invert:
            mask = 1 - mask
            
        return mask

    def create_preview(self, image, mask):
        # Create an RGBA preview with the mask as alpha channel
        if len(image.shape) == 2:
            image = image.unsqueeze(0).repeat(3, 1, 1)
        preview = image.clone()
        preview = torch.cat([preview, mask.unsqueeze(0)], dim=0)
        return preview

    def merge_segments(self, image, segments_to_merge_str, model_name, normalize_mask=True, 
                      binary_mask=False, resize_mode="bilinear", invert_mask=False, 
                      show_preview=True, blur_radius=5, dilation_radius=5, 
                      intensity=1.0, ceiling=1.0):
        show_on_node = False
    
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
            mode=resize_mode,
            align_corners=False if resize_mode != "nearest" else None,
        )

        pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()
        unique_segments = np.unique(pred_seg)

        # Handle empty segments string
        if not segments_to_merge_str.strip():
            segments_to_merge = []
        else:
            segments_to_merge = [int(s.strip()) for s in segments_to_merge_str.split(',') if s.strip()]

        merged_mask = np.zeros_like(pred_seg, dtype=np.float32)
        merged_segments = []

        for segment in unique_segments:
            if segment in segments_to_merge:
                mask = np.where(pred_seg == segment, 1, 0)
                merged_mask = np.maximum(merged_mask, mask)
                merged_segments.append(segment)

        # Convert to tensor and process
        merged_mask = torch.from_numpy(merged_mask)
        merged_mask = self.process_mask(
            merged_mask,
            normalize=normalize_mask,
            binary=binary_mask,
            invert=invert_mask,
            blur_radius=blur_radius,
            dilation_radius=dilation_radius,
            intensity=intensity,
            ceiling=ceiling
        )

        # Create preview if requested
        preview = self.create_preview(image[0], merged_mask) if show_preview else None

        # Apply mask to image
        merged_image = image[0].cpu().numpy() * merged_mask.numpy()[..., None]
        merged_image = torch.from_numpy(merged_image).unsqueeze(0)

        merged_segments_str = ','.join(map(str, merged_segments))
        if not merged_segments:
            merged_segments_str = "No segments selected"

        output_ui = {"images": [merged_image]} if show_on_node else {}

        return {"result": (merged_image, merged_mask, merged_segments_str, 
                preview if preview is not None else merged_image), 
                "ui": output_ui}

    
    
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