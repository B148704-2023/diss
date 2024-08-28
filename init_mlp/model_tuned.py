import torch
from torch import nn, optim
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_size, num_hidden_layers, output_size, dropout=0.30671519570891365):
        super(MLP, self).__init__()
        layers = []
        in_features = input_size

        # Calculate the step size for linear scaling
        step_size = (output_size - input_size) / (num_hidden_layers + 1)

        for i in range(num_hidden_layers):
            out_features = int(in_features + step_size)
            linear_layer = nn.Linear(in_features, out_features)
            layers.append(linear_layer)

            # Batch normalization
            layers.append(nn.BatchNorm1d(out_features))

            # activation
            layers.append(nn.Sigmoid())

            layers.append(nn.Dropout(p=dropout))

            in_features = out_features

        # Output layer
        layers.append(nn.Linear(in_features, output_size))
        #layers.append(nn.Softmax(dim=1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # Now the model directly returns the probabilities
        return self.model(x)


class DeconvolutionModel:
    def __init__(self, input_size, num_hidden_layers, output_size, device=None):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        print(f"Using {self.device} device")
        self.model = MLP(input_size, output_size, num_hidden_layers).to(self.device)
        self.loss_fn = WeightedErrorLoss()
        #self.loss_fn = WeightedErrorLoss() # example custom loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=2.1936548445907726e-05, weight_decay=0.06385399913015961)
        self.train_losses = []
        self.val_losses = []

    def fit(self, train_dataloader, val_dataloader, epochs, patience=100):
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

            print(f"Epoch {epoch+1}, Train loss: {avg_train_loss:>10f}, Validation loss: {avg_val_loss:>10f}")

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


class WeightedErrorLoss(nn.Module):
    def __init__(self, sum_penalty_weight=1.0):
        super(WeightedErrorLoss, self).__init__()
        self.sum_penalty_weight=sum_penalty_weight

    def forward(self, prediction, target):
        # Calculate the weight for each element based on the target value
        # Weight increases as the target moves away from 0
        mask = target != 0
        
        # Apply mask to predictions and targets
        masked_prediction = prediction[mask]
        masked_target = target[mask]

        zero_mask = target == 0

        # Compute Mean Squared Error on the non-zero elements
        zero_loss = torch.mean(prediction[zero_mask]**2)

        sum_penalty = torch.abs(prediction.sum(dim=1) - 1).mean()

        loss = torch.mean((masked_prediction - masked_target) ** 2)

        total_loss=loss+zero_loss+self.sum_penalty_weight * sum_penalty
        return total_loss