import matplotlib.pyplot as plt

def plot_training_validation_loss(train_losses, val_losses):
    plt.figure(figsize=(8, 6))
    plt.title('Training and Validation Loss')
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_training_validation_acc(train_accuracies, val_accuracies):
    plt.figure(figsize=(8, 6))
    plt.title('Training and Validation Accuracy')
    plt.plot(train_accuracies, label='Training Accuracy', color='blue')
    plt.plot(val_accuracies, label='Validation Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()