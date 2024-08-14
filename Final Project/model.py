import torch.nn as nn
import torch.nn.functional as F
import inspect

class SignalClassifier(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=32, out_channels=64, conv_kernel_size=5, pool_kernel_size=2, pool_stride=2, dropout=0.5, leak=0.1, input_size=8192):
        super(SignalClassifier, self).__init__()
        # recording parameters
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.conv_kernel_size = conv_kernel_size
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_stride
        self.dropout = dropout
        self.leak = leak
        self.input_size = input_size

        # Define the architectural layers
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=conv_kernel_size)
        self.conv2 = nn.Conv1d(in_channels=hidden_channels, out_channels=out_channels, kernel_size=conv_kernel_size)
        self.maxpool = nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_stride)
        
        # Calculate the size after conv and pooling layers
        conv_output_size = self.get_size(input_size, conv_kernel_size, pool_kernel_size, pool_stride)
        
        self.linear1 = nn.Linear(conv_output_size, 128)
        self.linear2 = nn.Linear(128, 4)  # output classes
    
    def get_size(self, input_size, conv_kernel_size, pool_kernel_size, pool_stride):
        # Helper function to calculate the output size after conv and pooling layers
        size = input_size
        size = (size - (conv_kernel_size - 1) - 1) // pool_stride + 1  # After conv1 and pool1
        size = (size - (conv_kernel_size - 1) - 1) // pool_stride + 1  # After conv2 and pool2
        return size * 64  # Multiply by the number of output channels of the last conv layer

    def params(self):
        init_params = inspect.signature(self.__init__).parameters
        return {name: getattr(self, name) for name in init_params if name != 'self'}

    def forward(self, x):
        # Connect the layers together and forward pass the input
        x = self.maxpool(F.relu(self.conv1(x)))
        x = nn.Dropout(self.dropout)(x)
        x = nn.LeakyReLU(self.leak)(x)

        x = self.pool(F.relu(self.conv2(x)))
        x = nn.Dropout(self.dropout)(x)
        x = nn.LeakyReLU(self.leak)(x)

        x = x.view(x.size(0), -1)  # Flatten layer
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x