import time
import torch
from transformers import BlipProcessor,AutoModel, BlipForConditionalGeneration,AutoFeatureExtractor,AutoModelForImageClassification,ViTFeatureExtractor, ViTForImageClassification, AutoModelForImageClassification
from PIL import Image
import numpy as np
from scipy.ndimage import binary_dilation
import torchvision.transforms as transforms
import os
import requests
from pathlib import Path
import folder_paths
import torchvision.models as models


class ImageCaptioningNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "caption"

    CATEGORY = "LexTools/ImageProcessing/Captioning"

    def __init__(self):
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")

    def caption(self, image):

        i = 255. * image[0].cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
      
        # Perform unconditional image captioning
        inputs = self.processor(img, return_tensors="pt").to("cuda")
        out = self.model.generate(**inputs)
        caption = self.processor.decode(out[0], skip_special_tokens=True)

        return (caption,)


class FoodCategoryClassifierNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"default": None}),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "classify_FoodCategory"
    CATEGORY = "LexTools/ImageProcessing/Classification"

    def __init__(self):
        self.feature_extractor = AutoFeatureExtractor.from_pretrained('Kaludi/food-category-classification-v2.0')
        self.model = AutoModelForImageClassification.from_pretrained('Kaludi/food-category-classification-v2.0')

    def classify_FoodCategory(self, image):
        i = 255. * image[0].cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        inputs = self.feature_extractor(images=img, return_tensors="pt")

        outputs = self.model(**inputs)
        proba = outputs.logits.softmax(1)

        # Get the top 5 class probabilities and their indices
        top_5_probs, top_5_indices = torch.topk(proba, 5)

        # Convert the probabilities and indices to lists
        top_5_probs = top_5_probs.tolist()[0]
        top_5_indices = top_5_indices.tolist()[0]

        # Get the labels from the model's configuration
        labels = self.model.config.id2label

        # Create a list of dictionaries with the class labels and probabilities
        results = [{"score": prob, "label": labels[idx]} for prob, idx in zip(top_5_probs, top_5_indices)]

        # Convert the list of dictionaries to a string and return it
        results_str = str(results)
        return [results_str]

class AgeClassifierNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"default": None}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "classify_age"
    CATEGORY = "LexTools/ImageProcessing/Classification"

    def __init__(self):
        self.feature_extractor = ViTFeatureExtractor.from_pretrained('nateraw/vit-age-classifier')
        self.model = ViTForImageClassification.from_pretrained('nateraw/vit-age-classifier')

    def classify_age(self, image):
        i = 255. * image[0].cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        inputs = self.feature_extractor(images=img, return_tensors="pt")

        outputs = self.model(**inputs)
        proba = outputs.logits.softmax(1)

        # Get the top 5 class probabilities and their indices
        top_5_probs, top_5_indices = torch.topk(proba, 5)

        # Convert the probabilities and indices to lists
        top_5_probs = top_5_probs.tolist()[0]
        top_5_indices = top_5_indices.tolist()[0]

        # Get the labels from the model's configuration
        labels = self.model.config.id2label

        # Create a list of dictionaries with the class labels and probabilities
        results = [{"score": prob, "label": labels[idx]} for prob, idx in zip(top_5_probs, top_5_indices)]

        # Convert the list of dictionaries to a string and return it
        results_str = str(results)
        return [results_str]
    

class ArtOrHumanClassifierNode:
    model = None
    feature_extractor = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "show_on_node": ("BOOLEAN", {"default": False}),
            },
        }
    OUTPUT_NODE = True

    RETURN_TYPES = ("FLOAT", "FLOAT")
    FUNCTION = "classify_image"
    CATEGORY = "LexTools/ImageProcessing/Classification"

    def __init__(self):
        if not ArtOrHumanClassifierNode.feature_extractor:
            ArtOrHumanClassifierNode.feature_extractor = ViTFeatureExtractor.from_pretrained("umm-maybe/AI-image-detector")

        if not ArtOrHumanClassifierNode.model:
            ArtOrHumanClassifierNode.model = AutoModelForImageClassification.from_pretrained("umm-maybe/AI-image-detector")

    def classify_image(self, image, show_on_node):
        i = 255. * image[0].cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        inputs = self.feature_extractor(images=img, return_tensors="pt")

        outputs = self.model(**inputs)
        proba = outputs.logits.softmax(1)

        # Get the probabilities for "artificial" and "human" classes
        artificial_prob = float(proba[0][0].item())
        human_prob = float(proba[0][1].item())

        output_ui =  {"text": [f"Artificial: {artificial_prob:.2%}\nHuman: {human_prob:.2%}"]} if show_on_node else {}

        return {"result": (artificial_prob, human_prob), "ui": output_ui}



    
class DocumentClassificationNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE", {"default": None})}}

    RETURN_TYPES = ("FLOAT", "STRING")
    FUNCTION = "classify"
    CATEGORY = "LexTools/ImageProcessing/Classification"

    def __init__(self):
        self.extractor = AutoFeatureExtractor.from_pretrained("DunnBC22/dit-base-Document_Classification-RVL_CDIP")
        self.model = AutoModelForImageClassification.from_pretrained("DunnBC22/dit-base-Document_Classification-RVL_CDIP")
        self.class_names = ['advertisement', 'budget', 'email', 'file_folder', 'form', 'handwritten', 'invoice', 'letter', 'memo', 'news_article', 'presentation', 'questionnaire', 'resume', 'scientific_publication', 'scientific_report', 'specification']

    def classify(self, image):
        # Convert the image tensor to a PIL Image
        image = Image.fromarray((image[0].numpy() * 255).astype(np.uint8).transpose(1, 2, 0))

        # Convert the image to the model's expected input format
        inputs = self.extractor(images=image, return_tensors="pt")

        # Perform the classification
        outputs = self.model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        predicted_class_index = torch.argmax(logits, dim=1).item()
        confidence_score = float(probabilities[0][predicted_class_index].item())

        # Get the class name
        predicted_class_name = self.class_names[predicted_class_index]

        return (confidence_score, predicted_class_name)

class NSFWClassifierNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"default": None}),
                "show_on_node": ("BOOLEAN", {"default": False}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
            },
        }
    OUTPUT_NODE = True

    RETURN_TYPES = ("STRING", "FLOAT", "FLOAT", "BOOLEAN", "BOOLEAN")  # Added boolean outputs
    RETURN_NAMES = ("Classification", "SFW Score", "NSFW Score", "Is SFW", "Is NSFW")
    FUNCTION = "classify_nsfw"
    CATEGORY = "LexTools/ImageProcessing/Classification"

    def __init__(self):
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("umairrkhn/fine-tuned-nsfw-classification")
        self.model = AutoModelForImageClassification.from_pretrained("umairrkhn/fine-tuned-nsfw-classification")

    def classify_nsfw(self, image, show_on_node, threshold):
        try:
            # Convert the image tensor to numpy array
            i = 255. * image[0].cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            # Process the image
            inputs = self.feature_extractor(images=img, return_tensors="pt")
            outputs = self.model(**inputs)
            probs = outputs.logits.softmax(1)[0]

            # Get probabilities for each class (model has 2 classes: SFW and NSFW)
            sfw_prob = float(probs[0].item())  # SFW
            nsfw_prob = float(probs[1].item())  # NSFW

            # Get the predicted class
            predicted_class_idx = probs.argmax().item()
            class_names = ["SFW", "NSFW"]
            predicted_class = class_names[predicted_class_idx]

            # Determine boolean states using threshold
            is_sfw = sfw_prob >= threshold
            is_nsfw = nsfw_prob >= threshold

            # Format the results string
            results = f"Predicted: {predicted_class}\n"
            results += f"SFW: {sfw_prob:.2%} ({'Yes' if is_sfw else 'No'})\n"
            results += f"NSFW: {nsfw_prob:.2%} ({'Yes' if is_nsfw else 'No'})"

            output_ui = {"text": [results]} if show_on_node else {}

            return {"result": (results, sfw_prob, nsfw_prob, is_sfw, is_nsfw), 
                    "ui": output_ui}

        except Exception as e:
            print(f"Error in NSFW classification: {str(e)}")
            return {"result": (str(e), 0.0, 0.0, False, False), 
                    "ui": {"text": [str(e)]} if show_on_node else {}}

class WatermarkDetectionNode:
    model = None  # Class-level model instance for caching
    transform = None  # Class-level transform for caching

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"default": None}),
                "show_on_node": ("BOOLEAN", {"default": False}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
            },
        }
    OUTPUT_NODE = True

    RETURN_TYPES = ("STRING", "FLOAT", "FLOAT", "BOOLEAN", "BOOLEAN")
    RETURN_NAMES = ("Classification", "Clean Score", "Watermark Score", "Is Clean", "Has Watermark")
    FUNCTION = "detect_watermark"
    CATEGORY = "LexTools/ImageProcessing/Classification"

    def download_model(self):
        # Create models directory if it doesn't exist
        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
        os.makedirs(models_dir, exist_ok=True)
        
        model_path = os.path.join(models_dir, "watermark_model.pt")
        
        # Download the model if it doesn't exist
        if not os.path.exists(model_path):
            print("Downloading watermark detection model...")
            url = "https://huggingface.co/qwertyforce/watermark_detection/resolve/main/model.pt"
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                with open(model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print("Model downloaded successfully")
            except Exception as e:
                print(f"Error downloading model: {str(e)}")
                # Try alternative URL from scenery_watermarks repo
                url = "https://huggingface.co/qwertyforce/scenery_watermarks/resolve/main/model.pt"
                print("Trying alternative model source...")
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                with open(model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print("Model downloaded successfully from alternative source")
            
        return model_path

    def __init__(self):
        if WatermarkDetectionNode.transform is None:
            # Standard EfficientNet preprocessing
            WatermarkDetectionNode.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        if WatermarkDetectionNode.model is None:
            try:
                # Create a new EfficientNet model
                base_model = models.efficientnet_b0(pretrained=False)
                # Modify the classifier for 2 classes
                base_model.classifier = torch.nn.Sequential(
                    torch.nn.Dropout(p=0.2, inplace=True),
                    torch.nn.Linear(in_features=1280, out_features=2, bias=True)
                )
                
                # Download and load the state dict
                model_path = self.download_model()
                state_dict = torch.load(model_path, map_location='cpu')
                
                # If it's a state dict, try to load it
                if isinstance(state_dict, dict):
                    try:
                        # Try direct loading
                        base_model.load_state_dict(state_dict)
                    except:
                        try:
                            # Try removing 'module.' prefix
                            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                            base_model.load_state_dict(new_state_dict)
                        except Exception as e:
                            print(f"Failed to load state dict: {str(e)}")
                            # If both attempts fail, just use the base model
                            pass
                else:
                    # If it's already a model, try to extract its state dict
                    try:
                        base_model.load_state_dict(state_dict.state_dict())
                    except:
                        print("Failed to load model state dict, using base model")
                
                WatermarkDetectionNode.model = base_model
                if torch.cuda.is_available():
                    WatermarkDetectionNode.model = WatermarkDetectionNode.model.cuda()
                WatermarkDetectionNode.model.eval()
                
            except Exception as e:
                print(f"Error loading watermark detection model: {str(e)}")
                raise

    def detect_watermark(self, image, show_on_node, threshold):
        try:
            # Convert the image tensor to PIL Image
            i = 255. * image[0].cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            
            # Ensure image is RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Preprocess the image
            img_tensor = self.transform(img).unsqueeze(0)
            if torch.cuda.is_available():
                img_tensor = img_tensor.cuda()

            # Get model predictions
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probs = torch.softmax(outputs, dim=1)[0]

            # Get probabilities for each class
            clean_prob = float(probs[0].item())  # Clean image
            watermark_prob = float(probs[1].item())  # Watermarked image

            # Get the predicted class
            predicted_class_idx = probs.argmax().item()
            class_names = ["Clean", "Watermarked"]
            predicted_class = class_names[predicted_class_idx]

            # Determine boolean states using threshold
            is_clean = clean_prob >= threshold
            has_watermark = watermark_prob >= threshold

            # Format the results string
            results = f"Predicted: {predicted_class}\n"
            results += f"Clean: {clean_prob:.2%} ({'Yes' if is_clean else 'No'})\n"
            results += f"Watermarked: {watermark_prob:.2%} ({'Yes' if has_watermark else 'No'})"

            output_ui = {"text": [results]} if show_on_node else {}

            return {"result": (results, clean_prob, watermark_prob, is_clean, has_watermark), 
                    "ui": output_ui}

        except Exception as e:
            print(f"Error in watermark detection: {str(e)}")
            return {"result": (str(e), 0.0, 0.0, False, False), 
                    "ui": {"text": [str(e)]} if show_on_node else {}}

NODE_CLASS_MAPPINGS = {
    "AgeClassifierNode": AgeClassifierNode,
    "FoodCategoryClassifierNode": FoodCategoryClassifierNode,
    "DocumentClassificationNode": DocumentClassificationNode,
    "ImageCaptioning": ImageCaptioningNode,
    "ArtOrHumanClassifierNode": ArtOrHumanClassifierNode,
    "NSFWClassifierNode": NSFWClassifierNode,
    "WatermarkDetectionNode": WatermarkDetectionNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageScaleToMin": "Image Scale To Min",
    "ImageCaptioning": "Image Captioning",
    "ArtOrHumanClassifierNode": "Art Or Human Classifier",
    "NSFWClassifierNode": "NSFW Classifier",
    "WatermarkDetectionNode": "Watermark Detector",
}