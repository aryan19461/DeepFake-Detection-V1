
# ===================== train_keras.py (Upgraded) =====================
import os, json, math, glob, random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB3

# ---------------- Config ----------------
SEED = 1337
IMG_SIZE = (300, 300)
BATCH_SIZE = 16
EPOCHS = 30
BASE_LR = 3e-4
USE_FOCAL = False         # set True to try Focal Loss
LABEL_SMOOTHING = 0.05
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'deepfake_detector_keras.h5')
DATA_DIR = 'data/dataset'
LABELS_JSON = 'labels.json'  # expects {"0":"real","1":"fake"}
AUTOTUNE = tf.data.AUTOTUNE

os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------- Reproducibility ----------------
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# ---------------- Utilities ----------------
def count_images(path):
    total = 0
    for r, d, files in os.walk(path):
        for f in files:
            if f.lower().endswith((".jpg",".jpeg",".png",".bmp",".webp")):
                total += 1
    return total

classes = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR,d))])
class_to_index = {c:i for i,c in enumerate(classes)}
print('Classes:', class_to_index)

# Write/refresh labels.json if missing
if not os.path.exists(LABELS_JSON):
    with open(LABELS_JSON, 'w') as f:
        json.dump({str(i):c for c,i in class_to_index.items()}, f)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    labels='inferred',
    label_mode='int',
    seed=SEED,
    validation_split=0.2,
    subset='training',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    labels='inferred',
    label_mode='int',
    seed=SEED,
    validation_split=0.2,
    subset='validation',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

# Compute class weights
label_counts = np.zeros(len(classes))
for _, y in train_ds.unbatch():
    label_counts[y.numpy()] += 1
class_weights = {i: float(np.sum(label_counts)/ (len(classes)*label_counts[i])) for i in range(len(classes))}
print('Class weights:', class_weights)

# Prefetch
train_ds = train_ds.shuffle(1024).prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# Data augmentation
augment = keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.1),
    layers.RandomBrightness(0.1),
], name='augment')

# Optional: MixUp
def sample_beta_distribution(batch_size, concentration=0.2, dtype=tf.float32):
    """Beta(a,b) via two Gammas; works with tensor batch_size inside tf.data map."""
    batch_size = tf.cast(batch_size, tf.int32)  # make it concrete int
    concentration = tf.cast(concentration, dtype)
    gamma1 = tf.random.gamma(shape=[batch_size], alpha=concentration, dtype=dtype)
    gamma2 = tf.random.gamma(shape=[batch_size], alpha=concentration, dtype=dtype)
    lam = gamma1 / (gamma1 + gamma2 + 1e-8)
    return lam

def mixup(batch_x, batch_y, alpha=0.2):
    batch_size = tf.shape(batch_x)[0]  # <-- tensor shape
    lam = sample_beta_distribution(batch_size, concentration=alpha, dtype=tf.float32)
    lam_x = tf.reshape(lam, [-1, 1, 1, 1])

    index = tf.random.shuffle(tf.range(batch_size))
    mixed_x = lam_x * batch_x + (1.0 - lam_x) * tf.gather(batch_x, index)

    y1 = tf.one_hot(batch_y, depth=len(classes), dtype=tf.float32)
    y2 = tf.one_hot(tf.gather(batch_y, index), depth=len(classes), dtype=tf.float32)
    mixed_y = tf.expand_dims(lam, -1) * y1 + tf.expand_dims(1.0 - lam, -1) * y2
    return mixed_x, mixed_y

# Wrap datasets to apply augment + (optional) mixup
def prepare(ds, training=True, use_mixup=True):
    def _map(x,y):
        x = tf.cast(x, tf.float32)/255.0
        if training:
            x = augment(x, training=True)
        if use_mixup and training:
            x, y = mixup(x, y)
            return x, y
        else:
            y = tf.one_hot(y, depth=len(classes))
            return x, y
    return ds.map(_map, num_parallel_calls=AUTOTUNE)

train_ds_p = prepare(train_ds, training=True, use_mixup=True)
val_ds_p = prepare(val_ds, training=False, use_mixup=False)

# Build model
inputs = layers.Input(shape=IMG_SIZE + (3,))
base = EfficientNetB3(include_top=False, input_tensor=inputs, weights='imagenet')
base.trainable = False  # warmup
x = layers.GlobalAveragePooling2D()(base.output)
x = layers.Dropout(0.35)(x)
outputs = layers.Dense(len(classes), activation='softmax', dtype='float32')(x)
model = keras.Model(inputs, outputs)

# Loss
if USE_FOCAL:
    # simple focal-loss wrapper
    def focal_loss(gamma=2., alpha=0.25):
        def _fl(y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            eps = tf.keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, eps, 1. - eps)
            cross_entropy = -y_true * tf.math.log(y_pred)
            weight = alpha * tf.pow(1 - y_pred, gamma)
            return tf.reduce_sum(weight * cross_entropy, axis=1)
        return _fl
    loss_fn = focal_loss()
else:
    loss_fn = keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING)

# LR schedule + compile
total_imgs = 0
for c in classes:
    total_imgs += count_images(os.path.join(DATA_DIR, c))
steps_per_epoch = max(1, total_imgs // BATCH_SIZE)
cosine = keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=BASE_LR, first_decay_steps=steps_per_epoch*5)
opt = keras.optimizers.Adam(learning_rate=cosine)
model.compile(optimizer=opt, loss=loss_fn, metrics=[keras.metrics.CategoricalAccuracy(name='acc'), keras.metrics.AUC(name='auc')])

# Callbacks
ckpt = keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor='val_auc', mode='max', save_best_only=True, verbose=1)
stop = keras.callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=10, restore_best_weights=True)

# Warmup (frozen base)
hist1 = model.fit(train_ds_p, validation_data=val_ds_p, epochs=5, class_weight=class_weights, callbacks=[ckpt, stop])

# Fine-tune: unfreeze top layers
base.trainable = True
for layer in base.layers[:-60]:  # unfreeze last ~60 layers
    layer.trainable = False
opt_ft = keras.optimizers.Adam(learning_rate=1e-4)
model.compile(optimizer=opt_ft, loss=loss_fn, metrics=[keras.metrics.CategoricalAccuracy(name='acc'), keras.metrics.AUC(name='auc')])

hist2 = model.fit(train_ds_p, validation_data=val_ds_p, epochs=EPOCHS, class_weight=class_weights, callbacks=[ckpt, stop])

print('Best model saved to:', MODEL_PATH)
# ================== end train_keras.py ==================
