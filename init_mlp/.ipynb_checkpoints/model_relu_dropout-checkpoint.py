import torch
from torch import nn, optim
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_size, output_size, num_hidden_layers, dropout_p):
        super(MLP, self).__init__()
        layers = []
        in_features = input_size

        # Calculate the step size for linear scaling
        step_size = (output_size - input_size) / (num_hidden_layers + 1)

        for i in range(num_hidden_layers):
            out_features = int(in_features + step_size)
            linear_layer = nn.Linear(in_features, out_features)
            layers.append(linear_layer)

            layers.append(nn.Dropout(p=dropout_p))

            # Batch normalization
            layers.append(nn.BatchNorm1d(out_features))

            # activation
            layers.append(nn.ReLU())
            in_features = out_features

        # Output layer
        layers.append(nn.Linear(in_features, output_size))
        #layers.append(nn.Softmax(dim=1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # Now the model directly returns the probabilities
        return self.model(x)

class DeconvolutionModelReluDropout:
    def __init__(self, input_size, output_size, num_hidden_layers, dropout_p, device=None):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        print(f"Using {self.device} device")
        self.model = MLP(input_size, output_size, num_hidden_layers, dropout_p).to(self.device)
        self.loss_fn = nn.MSELoss()
        #self.loss_fn = WeightedErrorLoss() # example custom loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.train_losses = []
        self.val_losses = []

    def fit(self, train_dataloader, val_dataloader, epochs=1, patience=100):
        best_val_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            for batch, (X, y) in enumerate(train_dataloader):
                X, y = X.to(self.device), y.to(self.device)

                # Compute prediction and loss
                pred = self.model(X)
                loss = self.loss_fn(pred, y)

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_dataloader)
            self.train_losses.append(avg_train_loss)
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for X, y in val_dataloader:
                    X, y = X.to(self.device), y.to(self.device)
                    pred = self.model(X)
                    loss = self.loss_fn(pred, y)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_dataloader)
            self.val_losses.append(avg_val_loss)

            print(f"Epoch {epoch+1}, Train loss: {avg_train_loss:>7f}, Validation loss: {avg_val_loss:>7f}")

            # Early stopping if model fails to improve
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                best_model_wts = self.model.state_dict()
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f'Early stopping on epoch {epoch+1}')
                    self.model.load_state_dict(best_model_wts)
                    break

    def evaluate(self, dataloader):
        self.model.eval()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss = 0

        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                test_loss += self.loss_fn(pred, y).item()

        test_loss /= num_batches
        print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")

    def predict(self, dataloader):
        self.model.eval()
        predictions = []
        actuals = []

        with torch.no_grad():
            for batch in dataloader:
                X, y = batch
                X = X.to(self.device)
                pred = self.model(X)
                predictions.append(pred.cpu().numpy())
                actuals.append(y.cpu().numpy())

        return np.concatenate(predictions), np.concatenate(actuals)

# trying a custom loss function
class WeightedErrorLoss(nn.Module):
    def __init__(self):
        super(WeightedErrorLoss, self).__init__()

    def forward(self, prediction, target):
        # Calculate the weight for each element based on the target value
        # Weight increases as the target moves away from 0
        weights = target  # As the target value is between 0 and 1, use it directly as weight
        squared_errors = (prediction - target) ** 2
        weighted_squared_errors = weights * squared_errors
        return weighted_squared_errors.mean()
