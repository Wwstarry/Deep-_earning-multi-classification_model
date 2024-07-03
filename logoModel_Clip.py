import os
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as transforms
import clip
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
from loguru import logger
from tqdm import tqdm
import multiprocessing
from multiprocessing import freeze_support
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
class LogoDataset(Dataset):
    def __init__(self, img_root, labels_csv, is_train, preprocess):
        self.img_root = img_root
        self.labels_csv = labels_csv
        self.is_train = is_train
        self.img_process = preprocess
        self.samples = []
        self.sam_labels = []

        # 读取CSV文件获取样本和标签
        df = pd.read_csv(labels_csv)
        for _, row in df.iterrows():
            img_path = os.path.join(self.img_root, row['Filename'])
            label = row['Label']
            self.samples.append(img_path)
            self.sam_labels.append(label)

        # 转换为token
        self.tokens = clip.tokenize(self.sam_labels)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        token = self.tokens[idx]
        # 加载图像
        image = Image.open(img_path).convert('RGB')
        # 对图像进行转换
        image = self.img_process(image)
        return image, token

# 设置路径
train_dir = './logo/train'
test_dir = './logo/test'
train_labels_csv = './train_labels.csv'
test_labels_csv = './test_labels.csv'

# 创建模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net, preprocess = clip.load("RN50", device=device, jit=False)

# 创建优化器和学习率调度器
optimizer = optim.Adam(net.parameters(), lr=1e-6, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# 创建损失函数
loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()

# 加载训练和测试数据集
train_dataset = LogoDataset(img_root=train_dir, labels_csv=train_labels_csv, is_train=True, preprocess=preprocess)
test_dataset = LogoDataset(img_root=test_dir, labels_csv=test_labels_csv, is_train=False, preprocess=preprocess)

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=0, pin_memory=True)

# 训练模型
num_epochs = 50
model_name = "logo_model"
ckpt_gap = 4  # 设置保存模型检查点的间隔


def accuracy(output, target):
    """Compute the accuracy of the model's output given the target labels."""
    _, pred = output.max(dim=1)
    correct = pred.eq(target).sum().item()
    total = target.size(0)
    return correct / total


if __name__ == '__main__':
    freeze_support()

    for epoch in range(num_epochs):
        scheduler.step()
        total_loss = 0
        phase = "train"
        dataloader = train_loader if phase == "train" else test_loader

        # 使用混合精度，占用显存更小
        with torch.cuda.amp.autocast(enabled=True):
            with tqdm(total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
                for images, label_tokens in dataloader:
                    images = images.to(device)
                    label_tokens = label_tokens.to(device)

                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == "train"):
                        logits_per_image, logits_per_text = net(images, label_tokens)
                        ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
                        cur_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
                        total_loss += cur_loss
                        if phase == "train":
                            cur_loss.backward()
                            optimizer.step()

                    pbar.set_postfix({"loss": cur_loss.item()})
                    pbar.update(1)

            epoch_loss = total_loss / len(dataloader)
            logger.info('{} Loss: {:.4f}'.format(phase, epoch_loss.item()))

            # Evaluate the model on the test set
            if phase == "train":
                net.eval()  # Set the model to evaluation mode
                with torch.no_grad():
                    total_accuracy = 0
                    for images, label_tokens in test_loader:
                        images = images.to(device)
                        label_tokens = label_tokens.to(device)
                        logits_per_image, logits_per_text = net(images, label_tokens)
                        total_accuracy += accuracy(logits_per_image, label_tokens)
                    avg_accuracy = total_accuracy / len(test_loader)
                    logger.info('Test Accuracy: {:.4f}'.format(avg_accuracy))
                net.train()  # Set the model back to training mode

            if epoch % ckpt_gap == 0:
                checkpoint_path = f"{model_name}_ckt.pth"
                checkpoint = {
                    'epoch': epoch,
                    'network': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Checkpoint {epoch} saved at {checkpoint_path}")

    # 保存最终模型
    torch.save(net.state_dict(), f"{model_name}_final.pth")
    logger.info(f"Final model saved as {model_name}_final.pth")
