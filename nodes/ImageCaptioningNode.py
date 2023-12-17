import time
import torch
from transformers import BlipProcessor,AutoModel, BlipForConditionalGeneration,AutoFeatureExtractor,AutoModelForImageClassification,ViTFeatureExtractor, ViTForImageClassification, AutoModelForImageClassification
from PIL import Image
import numpy as np
from scipy.ndimage import binary_dilation


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
        artificial_prob = proba[0][0].item()
        human_prob = proba[0][1].item()

        output_ui =  {"text": [artificial_prob]} if show_on_node else {}

        return {"result": (artificial_prob, human_prob), "ui": output_ui}



    
class DocumentClassificationNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE", {"default": None})}}

    RETURN_TYPES = ("INT", "STRING")
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
        predicted_class_index = torch.argmax(logits, dim=1).item()

        # Get the class name
        predicted_class_name = self.class_names[predicted_class_index]

        return (predicted_class_index, predicted_class_name)

NODE_CLASS_MAPPINGS = {
    "AgeClassifierNode": AgeClassifierNode,
    "FoodCategoryClassifierNode": FoodCategoryClassifierNode,
    "DocumentClassificationNode": DocumentClassificationNode,
    "ImageCaptioning": ImageCaptioningNode,
    "ArtOrHumanClassifierNode": ArtOrHumanClassifierNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageScaleToMin": "Image Scale To Min",
    "ImageCaptioning": "Image Captioning",
    "ArtOrHumanClassifierNode": "Art Or Human Classifier",
}