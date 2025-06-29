import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    r2_score,
)
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model, Sequential
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizerFast


def unpickle(file):
    import pickle  # noqa: S403

    with open(file, "rb") as fo:
        mydict = pickle.load(fo, encoding="bytes")  # noqa: S301
    return mydict


def create_text_embeddings(text_inputs, batch_size=16):
    text_arr = None
    total_inputs = len(text_inputs)
    for i in range(0, total_inputs, batch_size):
        inputs = text_inputs.tolist()[i : min(i + batch_size, total_inputs)]
        inputs = tokenizer(text=inputs, return_tensors="pt", padding=True, truncation=True)
        text_embeddings = model.get_text_features(**inputs)
        text_embeddings = text_embeddings.cpu().detach().numpy()
        if text_arr is None:
            text_arr = text_embeddings
        else:
            text_arr = np.concatenate((text_arr, text_embeddings), axis=0)
    print("Process of creating text embeddings completed", text_arr.shape, type(text_arr))  # noqa: T201
    return text_arr


def create_images_embeddings(images_input, batch_size=16):
    images_arr = None
    total_inputs = len(images_input)
    for i in range(0, total_inputs, batch_size):
        image_inputs = images_input[i : min(i + batch_size, total_inputs)]  # (1, 3, 224,224)
        images = processor(
            text=None,
            images=image_inputs.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1),
            do_rescale=False,
            return_tensors="pt",
        )["pixel_values"].to(device)
        images_embeddings = model.get_image_features(images)
        images_embeddings = images_embeddings.cpu().detach().numpy()
        if images_arr is None:
            images_arr = images_embeddings
        else:
            images_arr = np.concatenate((images_arr, images_embeddings), axis=0)
    print("Process of creating image embeddings completed", images_arr.shape, type(images_arr))  # noqa: T201
    return images_arr


def train_evaluation_knn_classifier(train_features, train_labels, test_features, test_labels, weights="uniform"):
    knn = KNeighborsClassifier(n_neighbors=3, weights=weights)
    knn.fit(train_features, train_labels)
    predictions = knn.predict(test_features)
    evaluation_reports = (confusion_matrix(test_labels, predictions), classification_report(test_labels, predictions))
    return knn, evaluation_reports


def create_text_inputs(row):
    return f"a photo of a {row['class_label']} which belongs to {row['superclass_label']}"


base_dir = os.path.dirname(os.path.abspath(__file__))

train = os.path.join(base_dir, "cifar-100-python", "train")
test = os.path.join(base_dir, "cifar-100-python", "test")
meta = os.path.join(base_dir, "cifar-100-python", "meta")

data = unpickle(train)
test = unpickle(test)
meta = unpickle(meta)

class_number = list(range(0, 100))
class_names = meta[b"fine_label_names"]  # class_labels

superclass_number = list(range(0, 20))
superclass_names = meta[b"coarse_label_names"]  # superclass_labels

# Mapping the class and superclass numbers to the corresponding labels' name based on the index
class_dict = dict(zip(class_number, class_names))
superclass_dict = dict(zip(superclass_number, superclass_names))

# Train Set with classes and superclasses labels
df = pd.DataFrame(data=data[b"data"], columns=list(range(1, 3073)))  # pixels
df["class_number"] = data[b"fine_labels"]  # class nums
df["superclass_number"] = data[b"coarse_labels"]
# Return the class and the superclass label names
df["class_label"] = df["class_number"].apply(lambda num: class_dict[num])
df["class_label"] = df["class_label"].apply(lambda value: value.decode("utf-8"))
df["superclass_label"] = df["superclass_number"].apply(lambda num: superclass_dict[num])
df["superclass_label"] = df["superclass_label"].apply(lambda value: value.decode("utf-8"))


# Test Set with classes and superclasses labels
df_test = pd.DataFrame(data=test[b"data"], columns=list(range(1, 3073)))  # pixels
df_test["class_number"] = test[b"fine_labels"]  # class_nums
df_test["superclass_number"] = test[b"coarse_labels"]

# Test set with the class and superclass label names
df_test["class_label"] = df_test["class_number"].apply(lambda num: class_dict[num])
df_test["class_label"] = df_test["class_label"].apply(lambda value: value.decode("utf-8"))
df_test["superclass_label"] = df_test["superclass_number"].apply(lambda num: superclass_dict[num])
df_test["superclass_label"] = df_test["superclass_label"].apply(lambda value: value.decode("utf-8"))

# Choose randomly ten different classes
random.seed(41)
chosen_classes = random.sample(population=class_number, k=10)
assert len(set(chosen_classes)) == 10
df = df[df["class_number"].isin(chosen_classes)]
df_test = df_test[df_test["class_number"].isin(chosen_classes)]

df["index"] = list(range(0, 5000))
df.set_index("index", inplace=True)
df_test["index"] = list(range(0, 1000))
df_test.set_index("index", inplace=True)

df.drop(["class_number", "superclass_number"], axis=1, inplace=True)
df_test.drop(["class_number", "superclass_number"], axis=1, inplace=True)

# Initial data for training
X_train = df.iloc[:, :-2].values / 255.0
y_train = df.iloc[:, -2].values

# Initial data for testing
X_test = df_test.iloc[:, :-2].values / 255.0
y_test = df_test.iloc[:, -2].values

# Autoencoder architecture
autoencoder = Sequential(
    [
        Dense(units=1024, activation="relu", input_shape=(3072,)),
        Dense(units=512, activation="relu"),
        Dense(units=256, activation="relu"),
        Dense(units=128, activation="relu"),
        Dense(units=256, activation="relu"),
        Dense(units=512, activation="relu"),
        Dense(units=1024, activation="relu"),
        Dense(units=3072, activation="sigmoid"),
    ]
)

autoencoder.compile(optimizer="adam", loss="mse")

# Using early stopping callback
early_stop = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=4)

# Fitting the model
autoencoder.fit(
    X_train[:4501],
    X_train[:4501],
    epochs=70,
    batch_size=8,
    validation_data=(X_train[4501:], X_train[4501:]),
    callbacks=[early_stop],
)
autoencoder_loss = pd.DataFrame(autoencoder.history.history)

initial_images_test = X_test.reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1))
autoencoded_images_test = autoencoder.predict(X_test)
autoencoded_images_test = autoencoded_images_test.reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1))

fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(10, 15))
for i in range(0, 5):
    # Initial images
    axes[i, 0].imshow(initial_images_test[i])
    axes[i, 0].set_title("Initial Image")
    axes[i, 0].axis("off")
    # After autoencoding
    axes[i, 1].imshow(autoencoded_images_test[i])
    axes[i, 1].set_title("Autoencoded Image")
    axes[i, 1].axis("off")
plt.show()


plt.figure(figsize=(10, 6))
sns.lineplot(data=autoencoder_loss, x=autoencoder_loss.index.astype(int), y=autoencoder_loss["loss"], label="loss")
sns.lineplot(
    data=autoencoder_loss, x=autoencoder_loss.index.astype(int), y=autoencoder_loss["val_loss"], label="val loss"
)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Hyper-parametrisation check")
plt.legend()
plt.show()

autoencoded_images_test = autoencoded_images_test.transpose(0, 3, 1, 2).reshape(1000, 3072)
print("Autoencoder's R-squared: ", r2_score(X_test, autoencoded_images_test))  # noqa: T201

# Keeping only the encoder part
encoder = Model(inputs=autoencoder.inputs, outputs=autoencoder.get_layer(index=4).output)
encoded_features_train = encoder.predict(X_train)
encoded_features_test = encoder.predict(X_test)

device = "cuda" if torch.cuda.is_available() else "cpu"

model_id = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_id).to(device)
tokenizer = CLIPTokenizerFast.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)

images_embeddings_train = create_images_embeddings(images_input=X_train, batch_size=16)
images_embeddings_test = create_images_embeddings(images_input=X_test, batch_size=16)

knn_raw_images = train_evaluation_knn_classifier(
    train_features=X_train,
    train_labels=y_train,
    test_features=X_test,
    test_labels=y_test,
)
print("Confusion Matrx: \n", knn_raw_images[1][0])  # noqa: T201
print("Classification Report: \n", knn_raw_images[1][1], "\n" * 3)  # noqa: T201

knn_encoded_images = train_evaluation_knn_classifier(
    train_features=encoded_features_train,
    train_labels=y_train,
    test_features=encoded_features_test,
    test_labels=y_test,
)
print("Confusion Matrx: \n", knn_raw_images[1][0])  # noqa: T201
print("Classification Report: \n", knn_raw_images[1][1], "\n" * 3)  # noqa: T201

knn_image_embeddings = train_evaluation_knn_classifier(
    train_features=images_embeddings_train,
    train_labels=y_train,
    test_features=images_embeddings_test,
    test_labels=y_test,
)
print("Confusion Matrx: \n", knn_image_embeddings[1][0])  # noqa: T201
print("Classification Report: \n", knn_image_embeddings[1][1], "\n" * 3)  # noqa: T201

# ZERO-SHOT ClIP EMBEDDING
df["text_input"] = df.apply(create_text_inputs, axis=1)
df_test["text_input"] = df_test.apply(create_text_inputs, axis=1)
clip_labels = df["text_input"].unique()
# Create text_embeddings
label_embeddings = create_text_embeddings(text_inputs=clip_labels, batch_size=10)
label_embeddings = label_embeddings / np.linalg.norm(label_embeddings, axis=1, keepdims=True)

# Normalize image embeddings
images_embeddings_train_normalized = images_embeddings_train / np.linalg.norm(
    images_embeddings_train, axis=1, keepdims=True
)
image_embeddings_test_normalized = images_embeddings_test / np.linalg.norm(
    images_embeddings_test, axis=1, keepdims=True
)
image_embeddings = np.concatenate((images_embeddings_train_normalized, image_embeddings_test_normalized), axis=0)
score = np.dot(label_embeddings, image_embeddings.T)  # (10x512) * (512x6000) -> (10x6000)
#  Each column in the new array is the interior product of each image with all text labels.
indices_of_closest_distances = np.argmax(score, axis=0)  #
predictions = []
ground_truth_labels = pd.concat((df, df_test), axis=0)["text_input"]
for i in indices_of_closest_distances:
    predictions.append(clip_labels[i])
predictions = np.array(predictions)

print(f"Confusion Matrix:\n {confusion_matrix(ground_truth_labels, predictions)}", "\n" * 3)  # noqa: T201
print(f"Classification Report:\n {classification_report(ground_truth_labels, predictions)}", "\n" * 3)  # noqa: T201
print(f"Accuracy score {accuracy_score(ground_truth_labels, predictions)}", "\n" * 3)  # noqa: T201
