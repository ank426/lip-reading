import torch.nn as nn
import torchvision.models as models

class LipReadingModel(nn.Module):
    def __init__(self, num_classes, hidden_size=512, dropout=0.5):
        super(LipReadingModel, self).__init__()

        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.features = mobilenet.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Freeze backbone
        for param in self.features.parameters():
            param.requires_grad = False

        self.feature_dim = 1280

        self.gru = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()

        x = x.view(batch_size * seq_len, c, h, w)
        x = self.features(x)
        x = self.avgpool(x)

        x = x.view(batch_size, seq_len, -1)

        output, _ = self.gru(x)

        output = output[:, -1, :]
        output = self.dropout(output)
        output = self.fc(output)

        return output
