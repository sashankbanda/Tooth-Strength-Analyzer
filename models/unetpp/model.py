import segmentation_models_pytorch as smp

def get_model_structural_segmentation(num_classes=4, encoder_name="resnet34", encoder_weights="imagenet"):
    """
    Returns a U-Net++ model for structural segmentation.
    
    Args:
        num_classes: Number of output classes (Background, Root, Bone, CEJ)
        encoder_name: Backbone encoder (e.g., 'resnet34', 'efficientnet-b0')
        encoder_weights: Pretrained weights for encoder (None for random initialization)
    """
    model = smp.UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=num_classes,
        activation=None # Return raw logits for CrossEntropyLoss, or handle in inference
    )
    return model
