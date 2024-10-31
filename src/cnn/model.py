import torch
from torch import nn
from torch.nn import functional as F

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, padding=1)
        self.max_pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.max_pool2 = nn.MaxPool2d(2)
        self.linear = nn.Linear(1280, 64)
        self.dropout = nn.Dropout(0.2)
        self.gru = nn.GRU(64, 32, bidirectional=True, num_layers=2, dropout=0.25)
        self.out = nn.Linear(64, num_classes + 1)

    def forward(self, images, labels=None, labels_mask=None):
        bs, c, h, w = images.size()
        # print(bs, c, h, w)
        x = F.relu(self.conv1(images))
        # print(x.size())
        x = self.max_pool1(x)
        # print(x.size())
        x = F.relu(self.conv2(x))
        # print(x.size())
        x = self.max_pool2(x) # 1, 64, 20, 195
        # print(x.size())
        x = x.permute(0, 3, 1, 2) # 1, 195, 64, 20
        # print(x.size())
        x = x.view(bs, x.size(1), -1) # 1, 195, 1280
        # print(x.size())
        x = self.linear(x)
        x = self.dropout(x) # 1, 195, 64
        # print(x.size())
        x, _ = self.gru(x)
        # print(x.size())
        x = self.out(x)
        # print(x.size())
        x = x.permute(1, 0, 2)
        # print(x.size())
        if labels is not None:
            log_softmax_values = F.log_softmax(x, 2)
            input_lengths = torch.full(
                size=(bs,), fill_value=log_softmax_values.size(0), dtype=torch.int32
            )
            # print(input_lengths)
            if labels_mask is not None:
                target_lengths = labels_mask.sum(dim=1, dtype=torch.int32)
            else:
                target_lengths = torch.full(
                    size=(bs,), fill_value=labels.size(1), dtype=torch.int32
                )
            # print(target_lengths)
            loss = nn.CTCLoss(blank=0)(
                log_softmax_values, labels, input_lengths, target_lengths
            )
            return x, loss
        return x, None

# cm = CNN(36)
# img = torch.randn(5, 3, 80, 780)
# label = torch.randint(1, 37, (5, 5))
# label[0][4] = -1
# label[0][3] = -1
# label_mask = label != -1
# x, loss = cm(img, label, label_mask)
# print(label_mask)