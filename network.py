import torch
import torch.nn as nn


class Network(nn.Module):
    """Create a PyTorch Neural Network with two hidden layers."""

    def __init__(self, input_size, h1_n_nodes, h2_n_nodes, output_size):
        super(Network, self).__init__()
        self.hidden1 = nn.Linear(input_size, h1_n_nodes)
        self.hidden2 = nn.Linear(h1_n_nodes, h2_n_nodes)
        self.out = nn.Linear(h2_n_nodes, output_size)

    def forward(self, x):
        self.hidden1_activations = torch.sigmoid(self.hidden1(x))
        self.hidden2_activations = torch.sigmoid(self.hidden2(self.hidden1_activations))
        self.out_activations = self.out(self.hidden2_activations)
        return self.out_activations

    def train_network(self, X, y):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        loss_function = nn.CrossEntropyLoss()

        for epoch in range(1000):
            y_pred = self(X)

            loss_score = loss_function(y_pred, y)

            optimizer.zero_grad()

            loss_score.backward()

            optimizer.step()

            if epoch % 100 == 0:
                max_value, prediction = torch.max(y_pred, 1)
                predicted_y = prediction.data.numpy()
                actual_y = y.data.numpy()
                accuracy = (predicted_y == actual_y).sum() / actual_y.size

    def get_activations(self):
        return [
            self.hidden1_activations.detach().numpy(),
            self.hidden2_activations.detach().numpy(),
            self.out_activations.detach().numpy()
        ]

    def evaluate_model(self, X, y):
        y_pred = self(X)
        max_value, prediction = torch.max(y_pred, 1)
        predicted_y = prediction.data.numpy()
        actual_y = y.data.numpy()
        num_correct_preds = (predicted_y == actual_y).sum()
        accuracy = num_correct_preds / actual_y.size
        print(f'Amount of correctly prediced pieces of data: [{num_correct_preds}], Accuracy: [{accuracy}]')
        return accuracy
