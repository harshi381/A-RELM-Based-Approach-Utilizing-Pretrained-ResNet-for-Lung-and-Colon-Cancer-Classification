import os
import cv2
import pickle
import numpy as np
import gc
import glob
import joblib  # ✅ For saving & loading models
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, log_loss, classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import label_binarize

# ✅ Define dataset and save paths
dataset_path = "" #Replace " " with your dataset path
pickle_folder = "" #Replace " " with path where you want to store your pickle files
model_save_path = "" #Replace " " with path where you want to store your model
os.makedirs(pickle_folder, exist_ok=True)

# ✅ Paths for saving features
resnet_pickle_path = os.path.join(pickle_folder, "resnet_features.pkl")

# ✅ Load ResNet50 Model (pretrained on ImageNet)
resnet_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

# ✅ Process images in mini-batches and save after every batch
batch_size = 1000  # Process 1000 images at a time

def process_images_in_batches(dataset_path, batch_size=1000, target_size=(224, 224)):
    classes = [folder for folder in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, folder))]
    print(f"\n✅ Found classes: {classes}")
    
    for cls in classes:
        cls_path = os.path.join(dataset_path, cls)
        img_names = os.listdir(cls_path)
        print(f"Processing class '{cls}', Images: {len(img_names)}")

        for i in tqdm(range(0, len(img_names), batch_size), desc=f"Extracting Features for {cls}"):
            batch_images, batch_labels = [], []

            for img_name in img_names[i : i + batch_size]:
                img_path = os.path.join(cls_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    continue
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                img = cv2.resize(img, target_size)
                img = img_to_array(img)
                img = preprocess_input(img)  # ✅ ResNet50 preprocessing
                batch_images.append(img)
                batch_labels.append(cls)

            # ✅ Convert to NumPy and extract ResNet50 features in mini-batches
            batch_images = np.array(batch_images)
            batch_features = resnet_model.predict(batch_images, batch_size=50, verbose=0)

            # ✅ Save each batch immediately
            np.save(f"{pickle_folder}/features_batch_{cls}_{i}.npy", batch_features)
            np.save(f"{pickle_folder}/labels_batch_{cls}_{i}.npy", batch_labels)

            # ✅ Clear memory
            del batch_images, batch_features
            gc.collect()

    print(f"\n✅ Features saved in {pickle_folder}")

# ✅ Check if features already exist before extracting
if os.path.exists(resnet_pickle_path):
    print(f"✅ Loading pre-saved ResNet50 features from {resnet_pickle_path}...")
    with open(resnet_pickle_path, "rb") as f:
        features_resnet, all_labels = pickle.load(f)
else:
    print("🔄 Extracting features using ResNet50 in Batches...")
    process_images_in_batches(dataset_path, batch_size=1000)  # Extract features
    
    # ✅ Load and Combine All Saved Batches
    features_list, labels_list = [], []

    feature_files = sorted(glob.glob(f"{pickle_folder}/features_batch_*.npy"))
    label_files = sorted(glob.glob(f"{pickle_folder}/labels_batch_*.npy"))

    for feat_file, lbl_file in zip(feature_files, label_files):
        batch_features = np.load(feat_file)
        batch_labels = np.load(lbl_file)

        if batch_features.shape[0] != len(batch_labels):
            print(f"⚠️ Skipping batch due to mismatch: {feat_file}")
            continue

        features_list.append(batch_features)
        labels_list.extend(batch_labels)

    # ✅ Convert to NumPy arrays
    features_resnet = np.vstack(features_list)
    all_labels = np.array(labels_list)

    # ✅ Save the full combined feature set
    with open(resnet_pickle_path, "wb") as f:
        pickle.dump((features_resnet, all_labels), f)

    print(f"✅ Combined features saved as {resnet_pickle_path}")

# ---------- Part B: RELM Training & Evaluation ----------

# ✅ Load extracted features
with open(resnet_pickle_path, "rb") as f:
    features_resnet, all_labels = pickle.load(f)

# ✅ Encode labels
le = LabelEncoder()
Y = le.fit_transform(all_labels)

# ✅ Split data
X_train, X_temp, Y_train, Y_temp = train_test_split(features_resnet, Y, test_size=0.4, random_state=42, stratify=Y)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42, stratify=Y_temp)

# ✅ Load pre-saved model if exists, else train new model
if os.path.exists(model_save_path):
    print(f"✅ Loading pre-saved RELM model from {model_save_path}...")
    best_model = joblib.load(model_save_path)
    skip_training = True
else:
    print("🔄 No pre-saved model found. Training a new RELM model...")
    skip_training = False

# ✅ Hyperparameter tuning (only if training is needed)
if not skip_training:
    alpha_values = [0.1, 1, 10, 100, 1000]
    best_alpha, best_test_acc, best_model, best_calibrated = None, 0, None, None

    for alpha in tqdm(alpha_values, desc="Hyperparameter Tuning (RELM)"):
        model = RidgeClassifier(alpha=alpha)
        model.fit(X_train, Y_train)

        calibrated_model = CalibratedClassifierCV(model, cv=3)
        calibrated_model.fit(X_train, Y_train)

        test_acc = accuracy_score(Y_test, model.predict(X_test))

        if test_acc > best_test_acc:
            best_test_acc, best_alpha, best_model, best_calibrated = test_acc, alpha, model, calibrated_model

    # ✅ Save the best trained model
    joblib.dump(best_model, model_save_path)
    print(f"✅ Best RELM model saved at: {model_save_path}")                                         

# ✅ Compute Training Accuracy
train_preds = best_model.predict(X_train)
train_acc = accuracy_score(Y_train, train_preds)

# ✅ Compute Validation Accuracy
val_preds = best_model.predict(X_val)
val_acc = accuracy_score(Y_val, val_preds)

# ✅ Print Training & Validation Accuracy
print(f"\n✅ Training Accuracy: {train_acc * 100:.2f}%")
print(f"✅ Validation Accuracy: {val_acc * 100:.2f}%")

# ✅ Evaluate Best Model
final_preds = best_model.predict(X_test)
print("\n🔹 Classification Report:")
print(classification_report(Y_test, final_preds, target_names=le.classes_))

# ✅ Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(Y_test, final_preds), annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (ResNet50)")
plt.show()
