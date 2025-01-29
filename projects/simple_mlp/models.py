import torch
import torch.nn as nn
class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size, number_of_hidden_layers,
        output_size):
        super(MyModel, self).__init__()

        layers = []
        for i in range(number_of_hidden_layers - 1):
            lin = nn.Linear(input_size, hidden_size)
            act = nn.ReLU()
            layers.append(lin)
            layers.append(act)
            input_size = hidden_size
            #We decided to make the last layers smaller than the first layers
            if i != number_of_hidden_layers-2:
                hidden_size = hidden_size // 2 #Halve the number of neurons for the next layer, except if it is the last layer

        layers.append(nn.Linear(input_size, hidden_size))
        self.layers = nn.Sequential(*layers)
        #self.layersE = nn.ModuleList(layers)
    def forward(self, x):
        out = self.layers(x)
        # out = x
        # for layer in self.layersE:
        #     out = layer(out)
        return out

if __name__ == "__main__":
    model = MyModel(10, 128,
                    8, 2)
    print(model)
    out = model(torch.rand(16, 10))
    print(out.shape)