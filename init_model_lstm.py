import torch
from torch import nn, optim
import numpy as np
from sparsemax import Sparsemax

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, device):
        super(LSTMModel, self).__init__()
        self.device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define the LSTM layer(s)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Define the LayerNorm (apply only once after LSTM)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Define the output layer
        self.output_layer = nn.Linear(hidden_size, output_size)
        
        # Define Sparsemax activation
        self.sparsemax = Sparsemax(dim=1)

    def init_hidden(self, batch_size):
        # Initialize hidden state and cell state for the LSTM
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)

        return h0, c0

        # Pass the input through the LSTM layers
    def forward(self, x):
        batch_size=x.size(0)
        h0, c0 = self.init_hidden(batch_size)
        
        out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Apply LayerNorm after LSTM output
        out = self.layer_norm(out[:, -1, :])  # Take the output of the last time step
        
        # Apply the output layer
        out = self.output_layer(out)

        # Apply Sparsemax to the output
        out = self.sparsemax(out)
        
        return out
        
    

class DeconvolutionModel:
    def __init__(self, input_size, hidden_size, output_size, num_layers, device=None):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        print(f"Using {self.device} device")
        self.model = LSTMModel(input_size, hidden_size, output_size, num_layers, device).to(self.device)
        self.loss_fn = RMSELoss()
        #self.loss_fn = WeightedErrorLoss() # example custom loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
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


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss


class CustomSparseLoss(nn.Module):
    def __init__(self):
        super(CustomSparseLoss, self).__init__()
        self.bce_loss = nn.CrossEntropyLoss()
        self.mae_loss = nn.L1Loss(reduction='none')

    def forward(self, prediction, target):
        # Mask for non-zero target values
        non_zero_mask = target != 0
        # Mask for zero target values
        zero_mask = target == 0

        # Apply BCE loss for non-zero values
        bce_loss_non_zero = self.bce_loss(prediction[non_zero_mask], target[non_zero_mask])
        
        # Apply MAE or BCE loss for zero values
        mae_loss_zero = self.mae_loss(prediction[zero_mask], target[zero_mask])
        
        # Combine the losses, and compute mean of each
        total_loss = torch.mean(bce_loss_non_zero) + torch.mean(mae_loss_zero)
        return total_loss