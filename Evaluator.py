import torch
import torch.nn as nn
import torchvision.models as models

'''===============================================================
1. Title:

DLP spring 2021 Lab7 classifier

2. Purpose:

For computing the classification accruacy.

3. Details:

The model is based on ResNet18 with only chaning the
last linear layer. The model is trained on iclevr dataset
with 1 to 5 objects and the resolution is the upsampled
64x64 images from 32x32 images.

It will capture the top k highest accuracy indexes on generated
images and compare them with ground truth labels.

4. How to use

You should call eval(images, labels) and to get total accuracy.
images shape: (batch_size, 3, 64, 64)
labels shape: (batch_size, 24) where labels are one-hot vectors
e.g. [[1,1,0,...,0],[0,1,1,0,...],...]

==============================================================='''


class Evaluator:
    def __init__(self, path: str) -> None:
        # modify the path to your own path
        self.classnum = 24
        checkpoint = torch.load(path)

        self.resnet18 = models.resnet18(pretrained=False)
        self.resnet18.fc = nn.Sequential(
            nn.Linear(512, self.classnum),
            nn.Sigmoid()
        )

        self.resnet18.load_state_dict(checkpoint['model'])
        self.resnet18 = self.resnet18.cuda()

        self.resnet18.eval()

    def compute_acc(self, out: torch.Tensor, onehot_labels: torch.Tensor) -> float:
        batch_size = out.shape[0]

        accuracy = 0
        total = 0

        for i in range(batch_size):
            k = int(onehot_labels[i].sum().item())
            total += k

            _, image_index = out[i].topk(k)
            _, label_index = onehot_labels[i].topk(k)

            for j in image_index:
                if j in label_index:
                    accuracy += 1

        return (accuracy / total)

    def eval(self, images: torch.Tensor, labels: torch.Tensor) -> float:
        with torch.no_grad():
            # your image shape should be (batch, 3, 64, 64)
            gen_images = self.resnet18(images)
            accuracy = self.compute_acc(gen_images.cpu(), labels.cpu())

            return accuracy
