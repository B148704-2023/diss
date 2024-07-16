import os 
import sys 

import torch
from torch import nn, optim

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import optuna.visualization as vis
import plotly.io as pio

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from abc import ABC, abstractmethod

class MLP(nn.Module):
    def __init__(self, input_size, num_hidden_layers, hidden_layer_size, output_size, activation, dropout=0):
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
            if activation == 'ReLU':
                layers.append(nn.ReLU())
            elif activation == 'LeakyReLU':
                layers.append(nn.LeakyReLU())
            elif activation == 'Tanh':
                layers.append(nn.Tanh())
            elif activation == 'Sigmoid':
                layers.append(nn.Sigmoid())

            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))

            in_features = out_features

        # Output layer
        layers.append(nn.Linear(in_features, output_size))
        #layers.append(nn.Softmax(dim=1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # Now the model directly returns the probabilities
        return self.model(x)

    def fit(self, train_loader, validation_loader, learning_rate, weight_decay, epochs, patience=10, device=None):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        print(f"Using {self.device} device")
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.loss_fn = nn.MSELoss()

        self.train_dataloader = train_loader
        self.val_dataloader = validation_loader

        train_losses=[] 
        val_losses = []

        best_val_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            for X, y in self.train_dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                loss = self.loss_fn(pred, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(self.train_dataloader)
            train_losses.append(avg_train_loss)

            self.model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for X, y in self.val_dataloader:
                    X, y = X.to(self.device), y.to(self.device)
                    pred = self.model(X)
                    loss = self.loss_fn(pred, y)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(self.val_dataloader)
            val_losses.append(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break

        return train_losses, val_losses


class DeconvolutionModel(ABC):
    DEFAULT_CONFIG = {}

    def __init__(self, train_loader, test_loader):
        self.train_loader=train_loader
        self.test_loader=test_loader
    
    @abstractmethod
    def fit(self, params, epochs=500, patience=10):
        pass

    @abstractmethod
    def predict(self, x):
        pass

    def parse_config(self, config):
        parsed_config = {}
        for key, default in self.DEFAULT_CONFIG.items():
            if key in config:
                parsed_config[key] = config[key]
            else:
                parsed_config[key] = default

        return parsed_config

    def update_data(self, train_loader, test_loader):
        """
        Update the training and testing data.

        :param x_train: The new training features.
        :param y_train: The new training targets.
        :param x_test: The new testing features.
        :param y_test: The new testing targets.
        """
        self.train_loader=train_loader
        self.test_loader=test_loader
        
    def plot_loss(self, prefix):
        """
        Saves the training and validation loss curves to a PDF.

        :param prefix: The prefix for the PDF filename.
        """
        with PdfPages(f'{prefix}_loss_curves.pdf') as pdf:
            # Create a plot with matplotlib for the loss curves
            fig, ax = plt.subplots()
            ax.plot(self.train_losses, label='Train Loss')
            ax.plot(self.val_losses, label='Validation Loss')
            ax.set_title('Train and Validation Loss Curves')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()

            # Save the matplotlib figure
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)


class SNV2LinFreqs(DeconvolutionModel):
    DEFAULT_CONFIG = {
        'num_hidden_layers': (1, 5),
        'hidden_layer_size': (16, 256),
        'dropout': (0, 0.5),
        'lr': (1e-5, 1e-1),
        'weight_decay': (1e-5, 1e-1),
        'epochs': 500,
        'patience': 10
    }
    def __init__(self, train_loader, test_loader, input_size, output_size, device=None):
        super().__init__(train_loader, test_loader)
        self.input_size = input_size
        self.output_size = output_size
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        print(f"Using {self.device} device")
        self.study = None
        self.best_params = None


    
    def _objective(self, trial, config):
        parsed_config = self.parse_config(config)

        num_hidden_layers = trial.suggest_int('num_hidden_layers', *parsed_config['num_hidden_layers'])
        hidden_layer_size = trial.suggest_int('hidden_layer_size', *parsed_config['hidden_layer_size'])
        activation = trial.suggest_categorical('activation', ['ReLU', 'Tanh', 'Sigmoid'])
        dropout = trial.suggest_float('dropout', *parsed_config['dropout'])
        weight_decay = trial.suggest_float('weight_decay', *parsed_config['weight_decay'])
        learning_rate = trial.suggest_float('lr', *parsed_config['lr'], log=True)
        epochs=parsed_config['epochs']
        patience=parsed_config['patience']

        model = MLP(input_size=self.input_size, 
                    num_hidden_layers=num_hidden_layers, 
                    hidden_layer_size=hidden_layer_size, 
                    output_size=self.output_size, 
                    activation=activation,
                   dropout=dropout).to(self.device)

        train_losses, val_losses = model.fit(
            self.train_loader, 
            self.test_loader, 
            learning_rate=learning_rate, 
            weight_decay=weight_decay,
            epochs=epochs,
            patience=patience
        )
        
        trial.set_user_attr('train_losses', train_losses)
        trial.set_user_attr('val_losses', val_losses)

        return val_losses[-1]

    def fit(self, params, epochs=500, patience=10):
        num_hidden_layers = params['num_hidden_layers']
        hidden_layer_size = params['hidden_layer_size']
        dropout = params['dropout']
        learning_rate = params['lr']
        weight_decay = params['weight_decay']
        activation = params['activation']
        
        self.model = MLP(input_size=self.input_size, 
                    output_size=self.output_size, 
                    num_hidden_layers=num_hidden_layers, 
                    hidden_layer_size=hidden_layer_size, 
                    activation=activation).to(self.device)

        self.train_losses, self.val_losses = self.model.fit(
            self.train_loader, 
            self.validation_loader, 
            learning_rate=learning_rate, 
            weight_decay=weight_decay,
            epochs=epochs,
            patience=patience)
        
        # return the final validation loss
        return self.val_losses[-1]
    
    
    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            predictions = self.model(x)
            
        return predictions.cpu().numpy()

    
    def optimize(self, n_trials=100, config={}, warmup_steps=10, pruner=True):
        if pruner:
            pruner = MedianPruner(n_warmup_steps=warmup_steps)
        else:
            pruner = None
        sampler = TPESampler(n_startup_trials=warmup_steps)

        self.study = optuna.create_study(direction='minimize', sampler=sampler, pruner=pruner)
        self.study.optimize(lambda trial: self._objective(trial, config), n_trials=n_trials)

        # Print results
        print('Number of finished trials:', len(self.study.trials))
        print('Best trial:')
        trial = self.study.best_trial
        print('Value: ', trial.value)
        print('Params: ')
        for key, value in trial.params.items():
            print(f'    {key}: {value}')

        self.best_params = trial.params
        return self.best_params

    
    def plot_trials(self, prefix):
        # Plot Optuna's built-in summary plots
        fig = optuna.visualization.plot_optimization_history(self.study)
        pio.write_image(fig, f'{prefix}_optimization_history.pdf')

        fig = optuna.visualization.plot_parallel_coordinate(self.study)
        pio.write_image(fig, f'{prefix}_parallel_coordinate.pdf')

        fig = optuna.visualization.plot_slice(self.study)
        pio.write_image(fig, f'{prefix}_slice.pdf')

        fig = optuna.visualization.plot_contour(self.study)
        pio.write_image(fig, f'{prefix}_contour.pdf')

        # Plot loss curves for each trial
        with PdfPages(f'{prefix}_loss_curves.pdf') as pdf:
            # Sort trials by their value in ascending order
            sorted_trials = sorted(self.study.trials, key=lambda t: t.value)
            for trial in sorted_trials:
                train_losses = trial.user_attrs['train_losses']
                val_losses = trial.user_attrs['val_losses']

                # Create a plot with matplotlib for the loss curves
                fig, ax = plt.subplots()
                ax.plot(train_losses, label='Train Loss')
                ax.plot(val_losses, label='Validation Loss')
                ax.set_title(f'Trial {trial.number}: Value={trial.value:.4f}, Params={trial.params}')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.legend()
                
                # Save the matplotlib figure
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)


    def save_model(self, file_path):
        torch.save(self.model.state_dict(), file_path)


    def load_model(self, file_path):
        self.model = MLP(input_size=self.input_size, 
                    num_hidden_layers=self.num_hidden_layers, 
                    hidden_layer_size=self.hidden_layer_size, 
                    output_size=self.output_size, 
                    activation=self.activation,
                    dropout=self.dropout)
        self.model.load_state_dict(torch.load(file_path))
    