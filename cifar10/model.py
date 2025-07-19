import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def call_bn(bn, x):
    return bn(x)

class CNN(nn.Module):
    def __init__(self, input_channel=3, n_outputs=10, dropout_rate=0.25, top_bn=False):
        super(CNN, self).__init__()
        self.dropout_rate = dropout_rate
        self.top_bn = top_bn

        # Convolutional layers
        self.c1 = nn.Conv2d(input_channel, 128, kernel_size=3, stride=1, padding=1)
        self.c2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.c3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.c4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.c5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.c6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.c7 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0)
        self.c8 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0)
        self.c9 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0)

        # Fully connected layer
        self.l_c1 = nn.Linear(128, n_outputs)

        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(256)
        self.bn7 = nn.BatchNorm2d(512)
        self.bn8 = nn.BatchNorm2d(256)
        self.bn9 = nn.BatchNorm2d(128)

    def forward(self, x):
        h = x
        h = F.leaky_relu(call_bn(self.bn1, self.c1(h)), negative_slope=0.01)
        h = F.leaky_relu(call_bn(self.bn2, self.c2(h)), negative_slope=0.01)
        h = F.leaky_relu(call_bn(self.bn3, self.c3(h)), negative_slope=0.01)
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = F.dropout2d(h, p=self.dropout_rate, training=self.training)

        h = F.leaky_relu(call_bn(self.bn4, self.c4(h)), negative_slope=0.01)
        h = F.leaky_relu(call_bn(self.bn5, self.c5(h)), negative_slope=0.01)
        h = F.leaky_relu(call_bn(self.bn6, self.c6(h)), negative_slope=0.01)
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = F.dropout2d(h, p=self.dropout_rate, training=self.training)

        h = F.leaky_relu(call_bn(self.bn7, self.c7(h)), negative_slope=0.01)
        h = F.leaky_relu(call_bn(self.bn8, self.c8(h)), negative_slope=0.01)
        h = F.leaky_relu(call_bn(self.bn9, self.c9(h)), negative_slope=0.01)
        
        # Replace `h.data` with `h` directly
        h = F.avg_pool2d(h, kernel_size=h.shape[2])

        h = h.view(h.size(0), -1)
        logit = self.l_c1(h)
        
        if self.top_bn:
            logit = call_bn(self.bn_c1, logit)
            
        return logit
if __name__ == "__main__":
    print("Starting script...")
    sample_input = torch.randn(1, 3, 32, 32)  # Simulated input
    print("Sample input created:", sample_input.shape)

    model = CNN(input_channel=3, n_outputs=10, dropout_rate=0.25, top_bn=False)
    print("Model initialized.")

    model.eval()  # Set model to evaluation mode
    output = model(sample_input)  # Run input through the model

    print("Output shape:", output.shape)
    print("Output tensor:", output)
