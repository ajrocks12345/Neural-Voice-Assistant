import matplotlib.pyplot as plt
import numpy as np

# Epochs
epochs = np.arange(10)

# Synthetic data approximating the provided image
train_loss = [0.96, 0.29, 0.13, 0.09, 0.08, 0.03, 0.05, 0.03, 0.02, 0.04]
val_loss = [0.27, 0.17, 0.11, 0.08, 0.06, 0.11, 0.16, 0.11, 0.16, 0.15]

train_acc = [0.83, 0.94, 0.97, 0.98, 0.981, 0.991, 0.984, 0.989, 0.992, 0.99]
val_acc = [0.951, 0.973, 0.976, 0.984, 0.989, 0.978, 0.978, 0.984, 0.981, 0.981]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left Subplot: Loss
ax1.plot(epochs, train_loss, label='Train_Loss', color='#348ABD') # Blueish
ax1.plot(epochs, val_loss, label='Validation_Loss', color='#F29E38') # Orangeish
ax1.set_title('Train_Loss & Validation_Loss', fontsize=16)
ax1.set_ylim([-0.02, 1.0])
ax1.legend()

# Right Subplot: Accuracy
ax2.plot(epochs, train_acc, label='Train_Accuracy', color='#348ABD')
ax2.plot(epochs, val_acc, label='Validation_Accuracy', color='#F29E38')
ax2.set_title('Train_Accuracy & Validation_Accuracy', fontsize=16)
ax2.set_ylim([0.82, 1.0])
ax2.legend()

plt.tight_layout()
plt.subplots_adjust(bottom=0.2)
fig.text(0.5, 0.05, 'Figure 21: Training loss, Validation Loss, Training Accuracy & Validation Accuracy', ha='center', fontsize=14, family='serif')

plt.savefig('C:/Users/Albin/.gemini/antigravity/brain/f0870f4f-0f4a-421f-accc-2dd328b481f8/training_graphs.png', dpi=300, bbox_inches='tight')
print("Graph saved successfully!")
