import os
import sys
import argparse
import pandas as pd
import numpy as np

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sparsemax import Sparsemax

import matplotlib.pyplot as plt
import seaborn as sns

device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, device):
        super(LSTMModel, self).__init__()

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
        #self.softmax = nn.Softmax(dim=1)

    def init_hidden(self, batch_size):
        # Initialize hidden state and cell state for the LSTM
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

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
        #out = self.softmax(out)
        
        return out
        
    

class DeconvolutionModel:
    def __init__(self, input_size, hidden_size, output_size, num_layers, lr, device):
        self.model = LSTMModel(input_size, hidden_size, output_size, num_layers, device).to(device)
        self.loss_fn = RMSELoss()
        #self.loss_fn = WeightedErrorLoss() # example custom loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        #self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5)
        self.train_losses = []
        self.val_losses = []

    def fit(self, train_dataloader, val_dataloader, epochs, patience=10):
        best_val_loss = float('inf')
        epochs_no_improve = 0
        

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            for batch, (X, y) in enumerate(train_dataloader):
                X, y = X.to(device), y.to(device)

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
                    X, y = X.to(device), y.to(device)
                    pred = self.model(X)
                    loss = self.loss_fn(pred, y)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_dataloader)
            self.val_losses.append(avg_val_loss)
            #self.scheduler.step(avg_val_loss)

            print(f"Epoch {epoch+1}, Train loss: {avg_train_loss:.10f}, Validation loss: {avg_val_loss:.10f}")

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
                X, y = X.to(device), y.to(device)
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
                X = X.to(device)
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

def main():

    parser = argparse.ArgumentParser(description="Training the Lineage Abundance Estimation LSTM on the Eddie computer cluster")
    parser.add_argument('--sample_size', type=int, help="Number of training examples to use, specifying training data file path")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training (default: 64)")
    parser.add_argument('--hidden_size', type=int, default=126, help="Number of neurons in the LSTM's hidden layer(s) (default:126)")
    parser.add_argument('--num_layers', type=int, default=1, help="Number of hidden layers in the LSTM (default: 1)")
    parser.add_argument('--learning_rate', type=float, default=0.0001, help="Learning rate for the LSTM model (default:0.0001)")

    args = parser.parse_args()

    n = args.sample_size
    batch_size = args.batch_size
    hidden_size = args.hidden_size
    num_layers = args.num_layers
    lr = args.learning_rate
    
    known_freqs_file = f'known_freqs_{n}.npy'
    snv_freqs_file = f'snv_freqs_{n}.npy'

    snv_freqs=np.load(snv_freqs_file)
    known_freqs=np.load(known_freqs_file)

    snv_freqs=torch.from_numpy(snv_freqs).to(device)
    known_freqs=torch.from_numpy(known_freqs).to(device)

    batch_size_=snv_freqs.shape[0]
    seq_len=1
    input_size = snv_freqs.shape[1]
    output_size = known_freqs.shape[1]

    snv_freqs=snv_freqs.view(batch_size_, seq_len, input_size)

    dataset = TensorDataset(snv_freqs, known_freqs)
    total_size = len(dataset)
    test_size = total_size // 10  # 10% for test
    validation_size = total_size // 10  # 10% for validation
    train_size = total_size - test_size - validation_size  # 80% train

    # Split the dataset
    train_dataset, validation_dataset, test_dataset = random_split(
        dataset, [train_size, validation_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = DeconvolutionModel(input_size, hidden_size, output_size, num_layers, lr, device)
    model.fit(train_loader, validation_loader, epochs=5000)

    # Evaluate the model
    model.evaluate(validation_loader)

    # Predict using the model
    predictions, actuals = model.predict(test_loader)

    results_folder = f'training_report_{num_layers}_of_{hidden_size}_with_lr_{lr}'
    os.makedirs(results_folder, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(model.train_losses, label='Training Loss')
    plt.plot(model.val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss Over Epochs')
    plt.savefig(os.path.join(results_folder, 'Train & Val Loss.png'), format='png')  # Save the current figure
    plt.close()

    # Plot predicted vs observed frequencies as a scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=predictions.flatten(), y=actuals.flatten())
    plt.xlabel('Predicted Frequencies')
    plt.ylabel('Actual Frequencies')
    plt.title('Predicted vs Actual Frequencies')
    plt.savefig(os.path.join(results_folder, 'Predicted vs Actuals.png'), format='png')  # Save the current figure
    plt.close()

if __name__ == "__main__":
    main()