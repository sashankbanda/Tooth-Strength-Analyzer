
import torch
import segmentation_models_pytorch as smp
import training.config as config
from training.dataset_loader import ToothStructureDataset
import training.utils as utils
import os

def train_one_epoch(model, optimizer, criterion, data_loader, device, epoch):
    model.train()
    running_loss = 0.0
    
    for i, (images, masks) in enumerate(data_loader):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if i % 10 == 0:
            print(f"Epoch [{epoch}], Step [{i}/{len(data_loader)}], Loss: {loss.item():.4f}")
            
    print(f"Epoch [{epoch}] Average Loss: {running_loss / len(data_loader):.4f}")

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Training on device: {device}")
    
    # Dataset
    # We rely on specific folder structure for segmentation
    # User needs to ensure dataset/segmentation/images and dataset/segmentation/masks are populated
    dataset = ToothStructureDataset(config.SEGMENTATION_IMAGES_DIR, config.SEGMENTATION_MASKS_DIR)
    
    if len(dataset) == 0:
        print(f"Warning: No images found in {config.SEGMENTATION_IMAGES_DIR}")
        # Proceeding only if user populates it later, but generally should return
        # return

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS
    )
    
    # Model: U-Net++ with ResNet34 backbone
    # Classes: 0: Background, 1: Root, 2: Bone
    n_classes = 3 
    model = smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=n_classes,
    )
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(config.NUM_EPOCHS):
        train_one_epoch(model, optimizer, criterion, data_loader, device, epoch)
        
        # Save checkpoint
        torch.save(model.state_dict(), os.path.join(config.CHECKPOINT_DIR, f"unetpp_epoch_{epoch}.pth"))
        print(f"Saved checkpoint for epoch {epoch}")

if __name__ == "__main__":
    main()
