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

class SignalDataset(Dataset):
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

    def save_checkpoint(self, epoch, optimizer, path='checkpoint.pth'):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(state, path)

    def load_checkpoint(self, optimizer, path='checkpoint.pth'):
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

        params = [tuple(self.model.params().items())]
        logger.info(f'Training initiated with the following parameters:'
                    f'\nModel Parameters: {params[0]}\n')
        
        start_time = time.time()
        try:
            for epoch in range(1, epochs + 1):
                self.epoch = epoch
                # ========================= training losses =========================
                self.model.train()
                loss_ct, counter = 0, 0
                for i, (batch, labels) in enumerate(self.trloader):
                    batch_start = time.time()
                    counter += 1

                    batch, labels = batch.to(self.device), labels.to(self.device)
                    batch = batch.unsqueeze(1) # <-- add channel dimension
                    optimizer.zero_grad()

                    outputs = self.model(batch)
                    batch_loss = lsfn(outputs, labels)
                    loss_ct += batch_loss.item()
                    absolute_loss += batch_loss.item()

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
                        if outliers:
                            if averaging:
                                batch_trlosses.append(avg_loss)  # <-- average loss of the interval
                            else:
                                batch_trlosses.append(batch_loss.item())
                        if not outliers and epoch > 1:
                            if averaging:
                                batch_trlosses.append(avg_loss)  # <-- average loss of the interval
                            else:
                                batch_trlosses.append(batch_loss.item())
                        else:
                            continue
                        loss_ct, counter = 0, 0  # reset for next interval

                        # ------------------------- FOR REAL-TIME PLOTTING ------------------------------------------------------
                        if live_plot:  # Plot losses and validation accuracy in real-time
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
                    batch_loss.backward()
                    optimizer.step()

                # ========================= validation losses =========================
                self.model.eval()
                with torch.no_grad():
                    tot_valoss = 0
                    for batch, labels in self.valoader:

                        batch, labels = batch.to(self.device), labels.to(self.device)
                        batch = batch.unsqueeze(1)  # <-- add channel dimension

                        outputs = self.model(batch)
                        batch_loss = lsfn(outputs, labels)

                        tot_valoss += batch_loss.item()

                    avg_val_loss = tot_valoss / len(self.valoader)
                    valosses.append(avg_val_loss)

                    elapsed_time = time.time() - start_time
                    minutes, seconds = divmod(int(elapsed_time), 60)
                    learning_rate = optimizer.param_groups[0]['lr']

                    val_log = f'({int(minutes)}m {int(seconds):02d}s) | VALIDATION (Epoch {epoch}/{epochs}) | LR: {learning_rate} | Loss: {avg_val_loss:.4f} |  Abs. Loss: {absolute_loss:.2f} -----------'
                    logger.info(val_log)
                
                # checkpoint
                self.save_checkpoint(epoch, optimizer, path='saved_model.pth')
                logger.info(f'Checkpoint saved for epoch {epoch}.')

            end_time = time.time()
            
        except KeyboardInterrupt:
            logger.warning("Training was interrupted by the user.")
            self.save_checkpoint(epoch, optimizer, path='saved_model.pth')
            logger.info(f'Checkpoint saved for epoch {epoch}.')

        except Exception as e:
            logger.error(f"An error has occurred: {e}", exc_info=True)
            self.save_checkpoint(epoch, optimizer, path='saved_model.pth')
            logger.info(f'Checkpoint saved for epoch {epoch}.')
            raise

        finally:
            try:
                end_time = time.time()

                torch.save(self.model.state_dict(), 'saved_model.pth')
                logger.info(f"Model saved as 'saved_model.pth'.")

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
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                inputs = inputs.unsqueeze(1)  # Add channel dimension
                outputs = self.model(inputs)
                loss = self.lsfn(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        avg_loss = running_loss / len(self.valoader)
        print(f'Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}')
        
        # return accuracy, avg_loss


    def confusion_matrix(self):
        from sklearn.metrics import confusion_matrix as cm

        self.model.eval()
        y_true = []
        y_pred = []

        with torch.no_grad():
            for inputs, labels in self.valoader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                inputs = inputs.unsqueeze(1)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        cm = cm(y_true, y_pred)
        classes = ['NR', 'SR', 'FR', 'Noise']
        cm_df = pd.DataFrame(cm, index=classes, columns=classes)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_df, annot=True, cmap='Blues', fmt='d', cbar=False)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()



'''
------------------------------------------------------------------------------------------------------------------------------------------
'''
import sys
import gc
import importlib.util
class GetModelImages: 
    def __init__(self, path, loader, num_images):
        self.path = path
        self.loader = loader
        self.num_images = num_images
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward_pass(self):
        with torch.no_grad():
            data_iter = iter(self.loader)
            images, _ = next(data_iter)
            images = images[:self.num_images].to(self.device)
            reconstruction_images, _, _ = self.model(images)
            return images.cpu(), reconstruction_images.cpu()
    
    def __enter__(self):
        sys.path.append(self.path)
        spec = importlib.util.spec_from_file_location('train', os.path.join(self.path, 'train.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        self.model = module.get_model().to(self.device)
        self.model.load_state_dict(torch.load(os.path.join(self.path, 'saved_model.pth')))
        # from train import model as model_class
        # self.model = model_class.to(self.device)
        # self.model.load_state_dict(torch.load(f'{self.path}/saved_model.pth'))
        self.model.eval()

        original, reconstructions = self.forward_pass()
        return original, reconstructions

    def __exit__(self, exc_type, exc_value, traceback):
        sys.path.remove(self.path)
        self.model.to('cpu')
        del self.model
        self.model = None
        gc.collect()
        torch.cuda.empty_cache()