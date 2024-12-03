import csv
import os, time
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import matplotlib.pyplot as plt

# Dataset and Transformations
class ResizeTransform:
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        img = T.Resize((self.size, self.size))(img)
        return img, target

class RoboflowDataset(Dataset):
    def __init__(self, root, folder='train', transforms=None):
        self.root = root
        self.transforms = transforms
        self.folder = folder
        if not folder in ['test', 'train', 'valid']:
            print("Invalid folder parameter. Must be 'valid', 'test' or 'train'. Using default train folder.") 
            self.folder = "train"

        self.imgs = sorted(os.listdir(os.path.join(root, f"{self.folder}/images")))
        self.annotations = sorted(os.listdir(os.path.join(root, f"{self.folder}/labels")))

        # Filter images with valid annotations
        self.data = []
        for img, ann in zip(self.imgs, self.annotations):
            ann_path = os.path.join(root, f"{self.folder}/labels", ann)
            if os.path.exists(ann_path):
                with open(ann_path, 'r') as f:
                    lines = f.readlines()
                if len(lines) > 0:  # Check if there are any annotations
                    self.data.append((img, ann))

        self.to_tensor = T.ToTensor()

    def __getitem__(self, idx):
        img_name, ann_name = self.data[idx]
        img_path = os.path.join(self.root, f"{self.folder}/images", img_name)
        ann_path = os.path.join(self.root, f"{self.folder}/labels", ann_name)

        img = Image.open(img_path).convert("RGB")
        img = self.to_tensor(img)

        # Parse YOLO annotations
        with open(ann_path, 'r') as f:
            lines = f.readlines()

        boxes = []
        labels = []
        for line in lines:
            label, x_center, y_center, width, height = map(float, line.strip().split())
            xmin = (x_center - width / 2) * img.shape[2]
            xmax = (x_center + width / 2) * img.shape[2]
            ymin = (y_center - height / 2) * img.shape[1]
            ymax = (y_center + height / 2) * img.shape[1]
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(int(label))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # Exclude entries with no boxes
        if boxes.numel() == 0:
            raise ValueError(f"Empty boxes for image {img_name}")

        target = {"boxes": boxes, "labels": labels}
        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.data)


# Collate Function
def collate_fn(batch):
    return tuple(zip(*batch))

# Model Setup
def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# Metric Evaluation
def evaluate_model(model, dataloader, device):
    model.eval()
    metric = MeanAveragePrecision(iou_thresholds=[0.5, 0.75, 0.95])  # mAP@50, @75, @50-95

    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            preds = [{
                "boxes": o["boxes"].cpu(),
                "scores": o["scores"].cpu(),
                "labels": o["labels"].cpu()
            } for o in outputs]
            gt = [{
                "boxes": t["boxes"].cpu(),
                "labels": t["labels"].cpu()
            } for t in targets]

            metric.update(preds, gt)

    results = metric.compute()
    print(f"mAP@50: {results['map_50']:.4f}, mAP@50-95: {results['map']:.4f}")
    return results


if __name__ == "__main__":
    torch.cuda.empty_cache()  # Free up memory after each epoch or batch

    # Dataset and DataLoader
    num_epochs = 100
    img_size = 800
    batch_size = 16
    learning_rate = 0.01
    dataset_path = "SG-Road-Signs-2"
    transforms = ResizeTransform(img_size)

    # Dataset and DataLoader for the train set
    train_dataset = RoboflowDataset(dataset_path, folder='train', transforms=transforms)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Dataset and DataLoader for the test set
    test_dataset = RoboflowDataset(dataset_path, folder='valid', transforms=transforms)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Model Initialization
    print("Initialising model...")
    num_classes = 22  # Number of classes + 1 for background
    model = get_model(num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Optimizer and Scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)

    # Training and testing mAP tracking
    train_mAP_50_95 = []
    test_mAP_50_95 = []

    # Save training stats
    filename = 'rcnn_training_stats.csv'
    file = open(filename, 'a', newline='')
    writer = csv.writer(file)
    if file.tell() == 0:
        writer.writerow(
            ['epoch', 'epoch_loss', 'cls_loss', 'boxreg_loss', 
            'train-mAP@50:', 'train-mAP@50-95',
            'test-mAP@50:', 'test-mAP@50-95',
            ])

    for epoch in range(num_epochs):
        print(f"\nTraining epoch {epoch+1}...")
        start_t = time.time()

        # Training
        model.train()
        epoch_loss = 0
        classification_loss = 0
        box_reg_loss = 0

        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()
            classification_loss += loss_dict['loss_classifier'].item()
            box_reg_loss += loss_dict['loss_box_reg'].item()

        print(
            f"Epoch {epoch+1}/{num_epochs}, Total Loss: {epoch_loss:.4f}, "
            f"Classification Loss: {classification_loss:.4f}, Box Regression Loss: {box_reg_loss:.4f}"
        )

        # Evaluate on train set
        print(f"Evaluating on train set for Epoch {epoch+1}/{num_epochs}...")
        train_results = evaluate_model(model, train_loader, device)
        train_mAP_50_95.append(train_results.get("map", 0.0))

        # Evaluate on test set
        print(f"Evaluating on test set for Epoch {epoch+1}/{num_epochs}...")
        test_results = evaluate_model(model, test_loader, device)
        test_mAP_50_95.append(test_results.get("map", 0.0))

        # Print train and test mAP@50-95
        print(f"Epoch {epoch+1}: Train mAP@50-95 = {train_mAP_50_95[-1]:.4f}, Test mAP@50-95 = {test_mAP_50_95[-1]:.4f}, Time taken = {round(time.time() - start_t,4)}")
        lr_scheduler.step(train_mAP_50_95[-1])  # Adjust learning rate based on train mAP@50-95

        writer.writerow([
            str(epoch+1), f"{epoch_loss:.4f}", f"{classification_loss:.4f}", f"{box_reg_loss:.4f}",
            f"{train_results['map_50']:.4f}", f"{train_results['map']:.4f}",
            f"{test_results['map_50']:.4f}", f"{test_results['map']:.4f}",
        ])

        torch.cuda.empty_cache() # clear memory after every epoch

    # save model
    print("Saving model...")
    torch.save(model.state_dict(), f'RCNN_{num_epochs}_{batch_size}.pth')

    file.close()

    # Generate Graph: mAP on train and test set over epochs
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_mAP_50_95, label="Train mAP@50-95")
    plt.plot(range(1, num_epochs + 1), test_mAP_50_95, label="Test mAP@50-95")
    plt.xlabel("Epoch")
    plt.ylabel("mAP@50-95")
    plt.title("mAP on Train and Test Set Over Epochs")
    plt.legend()
    plt.grid()
    plt.show()
