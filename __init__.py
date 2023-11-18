#from .nodes.SegGPT import segGPTNode
from .nodes import SegformerNode,ImageCaptioningNode,ImageProcessingNode

NODE_CLASS_MAPPINGS = {

    **SegformerNode.NODE_CLASS_MAPPINGS,
    **ImageCaptioningNode.NODE_CLASS_MAPPINGS,
    **ImageProcessingNode.NODE_CLASS_MAPPINGS,
}