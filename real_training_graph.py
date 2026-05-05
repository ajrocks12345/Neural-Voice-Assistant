import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

# 1. Load actual data
print("Loading data...")
df = pd.read_csv('backend/home_dataset_vectorized.csv')
X = df.drop(columns=['Disease']).values
y = df['Disease']
classes = y.unique()

X_train, X_val, y_train, y_val = train_test_split(X, y.values, test_size=0.2, random_state=42)

# 2. Iteratively train model to simulate "epochs"
train_loss = []
val_loss = []
train_acc = []
val_acc = []

# We'll use warm_start to add trees sequentially (similar to epochs in NN)
clf = RandomForestClassifier(n_estimators=0, warm_start=True, random_state=42)

# 10 'Epochs' (Adding 10 trees per epoch to reach 100)
epochs = np.arange(1, 11)

print("Starting training evaluation...")
for i in epochs:
    clf.n_estimators += 10
    clf.fit(X_train, y_train)
    
    # Accuracy
    y_train_pred = clf.predict(X_train)
    y_val_pred = clf.predict(X_val)
    train_acc.append(accuracy_score(y_train, y_train_pred))
    val_acc.append(accuracy_score(y_val, y_val_pred))
    
    # Loss (Log Loss requires probabilities)
    y_train_prob = clf.predict_proba(X_train)
    y_val_prob = clf.predict_proba(X_val)
    
    # Use clf.classes_ to ensure order matches
    t_loss = log_loss(y_train, y_train_prob, labels=clf.classes_)
    v_loss = log_loss(y_val, y_val_prob, labels=clf.classes_)
    train_loss.append(t_loss)
    val_loss.append(v_loss)
    
    print(f"Epoch {i} | Val Acc: {val_acc[-1]:.4f} | Val Loss: {val_loss[-1]:.4f}")

# Normalize/Cap log loss for visual clarity if it spikes wildly
train_loss = np.clip(train_loss, 0, 1.0)
val_loss = np.clip(val_loss, 0, 1.0)

# 3. Create Plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot settings
epochs_x = np.arange(10) # 0 to 9 to match reference style

# Left Subplot: Loss
ax1.plot(epochs_x, train_loss, label='Train_Loss', color='#348ABD') 
ax1.plot(epochs_x, val_loss, label='Validation_Loss', color='#F29E38') 
ax1.set_title('Train_Loss & Validation_Loss', fontsize=16)
ax1.set_ylim([-0.05, max(max(train_loss), max(val_loss)) * 1.1])
ax1.legend()

# Right Subplot: Accuracy
ax2.plot(epochs_x, train_acc, label='Train_Accuracy', color='#348ABD')
ax2.plot(epochs_x, val_acc, label='Validation_Accuracy', color='#F29E38')
ax2.set_title('Train_Accuracy & Validation_Accuracy', fontsize=16)
ax2.set_ylim([min(min(train_acc), min(val_acc)) * 0.95, 1.01])
ax2.legend()

plt.tight_layout()
plt.subplots_adjust(bottom=0.2)
fig.text(0.5, 0.05, 'Figure 21: Real Training vs Validation Metrics (MediBot Random Forest)', ha='center', fontsize=14, family='serif')

save_path = 'C:/Users/Albin/.gemini/antigravity/brain/f0870f4f-0f4a-421f-accc-2dd328b481f8/real_training_metrics.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Graph saved successfully to {save_path}")
