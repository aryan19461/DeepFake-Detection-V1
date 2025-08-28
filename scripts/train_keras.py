import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers, callbacks

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "deepfake_detector_keras.h5")

IMG_SIZE = (128, 128)
BATCH = 32
EPOCHS = 5  # increase for real training

def make_model(input_shape=(128,128,3)):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer=optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return model

if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        raise SystemExit(f"Dataset folder not found: {DATA_DIR}. Put images under real/ and fake/ subfolders.")
    train_datagen = ImageDataGenerator(
        rescale=1/255.0,
        validation_split=0.2,
        horizontal_flip=True,
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.1,
    )

    train_gen = train_datagen.flow_from_directory(
        DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH, class_mode="binary", subset="training"
    )

    val_gen = train_datagen.flow_from_directory(
        DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH, class_mode="binary", subset="validation"
    )

    model = make_model((IMG_SIZE[0], IMG_SIZE[1], 3))
    model.summary()

    ckpt = callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True, mode="max")
    es = callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True)

    model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=[ckpt, es])

    model.save(MODEL_PATH)
    print("Model saved at:", MODEL_PATH)