import os
from re import search
import numpy as np
import pandas as pd
import seaborn as sns
from astropy.io import fits
import astropy.visualization as vis
import matplotlib.pyplot as plt
import time
from IPython.display import clear_output

import torch
from torch.utils.data import Dataset, DataLoader
import logging

class SignalDataset(Dataset): # custom dataset class to build for each strain
    def __init__(self, signals, signal_type):
        data_np = np.array(signals[signal_type])
        self.data = torch.tensor(data_np, dtype=torch.float32)
        self.labels = torch.tensor([0 if signal_type == 'NR' else 1 if signal_type == 'SR' else 2 if signal_type == 'FR' else 3 for _ in range(len(data_np))], dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
        

class experiment:
    def __init__(self, model, trloader, valoader, batch_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.trloader = trloader
        self.valoader = valoader
        # self.teloader = teloader
        self.batch_size = batch_size
        
        # self.x_dim = self.trloader.dataset[0][0].size()[1]*self.trloader.dataset[0][0].size()[2]
        self.train_size = len(self.trloader.dataset)

    def save_checkpoint(self, epoch, optimizer, path='checkpoint.pth'): # for saving the model state
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(state, path)

    def load_checkpoint(self, optimizer, path='checkpoint.pth'): # for going back to a previous training session
        if os.path.isfile(path):
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1
            return start_epoch
        else:
            raise FileNotFoundError(f"No checkpoint found at '{path}'")
    
    def train(self, optimizer, lsfn, epochs, live_plot=False, outliers=True, view_interval=100, averaging=True):
        self.lsfn = lsfn
        # ========================== Logger Configuration ==========================
        torch.backends.cudnn.benchmark = True
        torch.set_printoptions(profile="full")

        self.timestamp = time.strftime('%m-%d-%y__%H-%M-%S', time.localtime())

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%m/%d/%y %H:%M:%S')

        # file handler
        file_handler = logging.FileHandler(f'./Training Logs/{self.timestamp}.log')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # ---------------------------------------------------------------------------
        valosses = []  # <-- per epoch
        batch_trlosses = []  # <-- per batch
        batch_ints = self.train_size / (self.batch_size * view_interval)
        batch_times = []
        absolute_loss = 0
        self.epoch = 0

        # log the model parameters
        params = [tuple(self.model.params().items())]
        logger.info(f'Training initiated with the following parameters:'
                    f'\nModel Parameters: {params[0]}\n')
        
        start_time = time.time()
        try:
            for epoch in range(1, epochs + 1):
                self.epoch = epoch
                # ========================= training losses =========================
                self.model.train() # <-- set the model to training mode
                loss_ct, counter = 0, 0
                for i, (batch, labels) in enumerate(self.trloader): # iterate over the training loader
                    batch_start = time.time()
                    counter += 1

                    batch, labels = batch.to(self.device), labels.to(self.device)
                    batch = batch.unsqueeze(1) # <-- add channel dimension
                    optimizer.zero_grad() # zero the gradients

                    # forward pass
                    outputs = self.model(batch)
                    batch_loss = lsfn(outputs, labels)
                    loss_ct += batch_loss.item()
                    absolute_loss += batch_loss.item()

                    # log the batch loss
                    batch_time = time.time() - batch_start
                    elapsed_time = time.time() - start_time
                    minutes, seconds = divmod(int(elapsed_time), 60)
                    learning_rate = optimizer.param_groups[0]['lr']

                    batch_log = f'({int(minutes)}m {int(seconds):02d}s) | [{epoch}/{epochs}] Batch {i} ({batch_time:.3f}s) | LR: {learning_rate} | Loss: {batch_loss.item():.4f} | Abs. Loss: {absolute_loss:.2f}'
                    logger.info(batch_log)
                    batch_times.append(batch_time)

                    # ------------------------- Recording Loss ------------------------------------------------------
                    if (i + 1) % view_interval == 0 or i == len(self.trloader) - 1:  # <-- plot for every specified interval of batches (and also account for the last batch)
                        avg_loss = loss_ct / counter
                        if outliers: # include all losses in the plot
                            if averaging:
                                batch_trlosses.append(avg_loss)  # <-- average loss of the interval
                            else:
                                batch_trlosses.append(batch_loss.item())
                        if not outliers and epoch > 1: # exclude outliers from starting point of training
                            if averaging:
                                batch_trlosses.append(avg_loss)  # <-- average loss of the interval
                            else:
                                batch_trlosses.append(batch_loss.item())
                        else:
                            continue
                        loss_ct, counter = 0, 0  # reset for next interval

                        # ------------------------- FOR REAL-TIME PLOTTING ------------------------------------------------------
                        if live_plot:  # Plot losses and validation accuracy in real-time (when specified)
                            fig, ax = plt.subplots(figsize=(12, 5))
                            clear_output(wait=True)
                            ax.clear()

                            ax.set_title(f'Performance (Epoch {epoch}/{epochs})', weight='bold', fontsize=15)
                            ax.plot(list(range(1, len(batch_trlosses) + 1)), batch_trlosses,
                                    label=f'Training Loss \nLowest: {min(batch_trlosses):.3f} \nAverage: {np.mean(batch_trlosses):.3f} \n',
                                    linewidth=3, color='blue', marker='o', markersize=3)
                            if len(valosses) > 0:
                                ax.plot([i * batch_ints for i in range(1, len(valosses) + 1)], valosses,
                                        label=f'Validation Loss \nLowest: {min(valosses):.3f} \nAverage: {np.mean(valosses):.3f}',
                                        linewidth=3, color='gold', marker='o', markersize=3)
                            ax.set_ylabel("Loss")
                            ax.set_xlabel(f"Batch Iterations (per {view_interval} batches)")
                            ax.set_xlim(1, len(batch_trlosses) + 1)
                            ax.legend(title=f'Absolute loss: {round(absolute_loss, 3)}', bbox_to_anchor=(1, 1), loc='upper right')

                            plt.show(block=False)
                    # -------------------------------------------------------------------------------
                    batch_loss.backward() # backpropagate the loss
                    optimizer.step() # update the model parameters

                # ========================= validation losses =========================
                self.model.eval() # <-- set the model to evaluation mode
                with torch.no_grad(): # <-- disable gradient calculation
                    tot_valoss = 0 
                    for batch, labels in self.valoader: # iterate over the validation loader

                        batch, labels = batch.to(self.device), labels.to(self.device)
                        batch = batch.unsqueeze(1)  # <-- add channel dimension

                        outputs = self.model(batch) # get the model predictions
                        batch_loss = lsfn(outputs, labels) # calculate the loss

                        tot_valoss += batch_loss.item() # update the total loss

                    # calculate the average loss
                    avg_val_loss = tot_valoss / len(self.valoader)
                    valosses.append(avg_val_loss)

                    # log the validation loss
                    elapsed_time = time.time() - start_time
                    minutes, seconds = divmod(int(elapsed_time), 60)
                    learning_rate = optimizer.param_groups[0]['lr']

                    val_log = f'({int(minutes)}m {int(seconds):02d}s) | VALIDATION (Epoch {epoch}/{epochs}) | LR: {learning_rate} | Loss: {avg_val_loss:.4f} |  Abs. Loss: {absolute_loss:.2f} -----------'
                    logger.info(val_log)
                
                # checkpoint
                self.save_checkpoint(epoch, optimizer, path='saved_model.pth')
                logger.info(f'Checkpoint saved for epoch {epoch}.')
                clear_output(wait=True)

            end_time = time.time()
            
        except KeyboardInterrupt: # <-- when training process is interrupted
            logger.warning("Training was interrupted by the user.")
            self.save_checkpoint(epoch, optimizer, path='saved_model.pth')
            logger.info(f'Checkpoint saved for epoch {epoch}.')

        except Exception as e: # <-- catch any other exceptions
            logger.error(f"An error has occurred: {e}", exc_info=True)
            self.save_checkpoint(epoch, optimizer, path='saved_model.pth')
            logger.info(f'Checkpoint saved for epoch {epoch}.')
            raise

        finally:
            try:
                end_time = time.time() # <-- end time for the training process

                # Save the model
                torch.save(self.model.state_dict(), 'saved_model.pth')
                logger.info(f"Model saved as 'saved_model.pth'.")

                # log the training summary
                minutes, seconds = divmod(end_time - start_time, 60)
                logger.info(
                    '\n==========================================================================================='
                    '\n===========================================================================================\n'
                   f'\nModel Parameters: {params[0]}\n'
                   f'\nCompleted Epochs: {self.epoch}/{epochs} | Avg Tr.Loss: {np.mean(batch_trlosses):.3f} | Absolute Loss: {absolute_loss:.3f}'
                   f'\nTotal Training Time: {int(minutes)}m {int(seconds):02d}s | Average Batch Time: {np.mean(batch_times):.3f}s'
                )
                
                # Final plot to account for all tracked losses in an epoch
                fig, ax = plt.subplots(figsize=(12, 5))
                clear_output(wait=True)
                ax.clear()

                ax.set_title(f'Performance (Epoch {self.epoch}/{epochs})', weight='bold', fontsize=15)
                ax.plot(list(range(1, len(batch_trlosses) + 1)), batch_trlosses,
                        label=f'Training Loss \nLowest: {min(batch_trlosses):.3f} \nAverage: {np.mean(batch_trlosses):.3f} \n',
                        linewidth=3, color='blue', marker='o', markersize=3)
                if len(valosses) > 0:
                    ax.plot([i * batch_ints for i in range(1, len(valosses) + 1)], valosses,
                            label=f'Validation Loss \nLowest: {min(valosses):.3f} \nAverage: {np.mean(valosses):.3f}',
                            linewidth=3, color='gold', marker='o', markersize=3)
                ax.set_ylabel("Loss")
                ax.set_xlabel(f"Batch Iterations (per {view_interval} batches)")
                ax.set_xlim(1, len(batch_trlosses) + 1)
                ax.legend(title=f'Absolute loss: {round(absolute_loss, 3)}', bbox_to_anchor=(1, 1), loc='upper right')

                plt.show(block=False)
                plt.savefig(f"./Loss Plots/{self.timestamp}.png", bbox_inches='tight')

            except Exception as e:
                logger.error(f"An error has occurred: {e}", exc_info=True)
                raise

        return absolute_loss


    def evaluate(self):
        self.model.eval()  # Set the model to evaluation mode
        correct = 0
        total = 0
        running_loss = 0.0

        with torch.no_grad():  # Disable gradient calculation
            for inputs, labels in self.valoader:
                inputs, labels = inputs.to(self.device), labels.to(self.device) # Move to GPU
                inputs = inputs.unsqueeze(1)  # Add channel dimension
                outputs = self.model(inputs) # Get the model predictions
                loss = self.lsfn(outputs, labels) # Calculate the loss
                running_loss += loss.item() # Update the running loss
                _, predicted = torch.max(outputs, 1) # Get the predicted labels
                total += labels.size(0) # Update the total count
                correct += (predicted == labels).sum().item() # Update the correct count

        # Calculate the accuracy and average loss
        accuracy = correct / total
        avg_loss = running_loss / len(self.valoader)
        print(f'Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}')
        
        # return accuracy, avg_loss


    def confusion_matrix(self):
        from sklearn.metrics import confusion_matrix as cm

        # Set the model to evaluation mode
        self.model.eval()
        y_true = []
        y_pred = []

        with torch.no_grad(): # Disable gradient calculation
            # Iterate over the validation loader
            for inputs, labels in self.valoader:
                inputs, labels = inputs.to(self.device), labels.to(self.device) # Move to GPU
                inputs = inputs.unsqueeze(1) # Add channel dimension
                outputs = self.model(inputs) # Get the model predictions
                _, predicted = torch.max(outputs, 1) # Get the predicted labels

                # Append the true and predicted labels to the lists
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        # Generate the confusion matrix
        cm = cm(y_true, y_pred)
        classes = ['NR', 'SR', 'FR', 'Noise']
        cm_df = pd.DataFrame(cm, index=classes, columns=classes)
        
        # Plot the confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_df, annot=True, cmap='Blues', fmt='d', cbar=False)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()


