
# ===================== explain_utils.py =====================
import numpy as np, cv2, tensorflow as tf
from tensorflow import keras

# Grad-CAM for the top predicted class (or a target index)
def grad_cam(model, img_bgr, target_size=(300,300), target_class=None, layer_name=None):
    # Preprocess
    img = cv2.resize(img_bgr, target_size)
    x = img.astype('float32')/255.0
    x = np.expand_dims(x, 0)

    # Pick a conv layer if not provided
    if layer_name is None:
        # pick the last conv layer heuristically
        for l in reversed(model.layers):
            if isinstance(l, tf.keras.layers.Conv2D):
                layer_name = l.name; break
        if layer_name is None:
            # Fallback: try the third last layer
            layer_name = model.layers[-3].name

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x)
        if target_class is None:
            target_class = tf.argmax(preds[0])
        class_channel = preds[:, target_class]
    grads = tape.gradient(class_channel, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    # Resize & overlay
    heatmap_resized = cv2.resize(heatmap, (img_bgr.shape[1], img_bgr.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr, 0.6, heatmap_color, 0.4, 0)

    return heatmap_resized, overlay

# Lightweight artifact heuristics (for a short textual "why")
def artifact_reasons(bgr, heatmap):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # 1) High-frequency artifacts
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    # 2) Edge halos / boundary inconsistencies
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_mag = np.mean(np.sqrt(sobelx**2 + sobely**2))
    # 3) Where Grad-CAM focuses (e.g., mouth/eyes area proxy)
    h, w = heatmap.shape
    eye_mouth_band = heatmap[int(0.25*h):int(0.7*h), int(0.2*w):int(0.8*w)]
    focus_ratio = float(np.mean(eye_mouth_band > 0.5))

    reasons = []
    if lap_var > 2000:
        reasons.append("Unnatural high‑frequency noise patterns detected (compression or GAN artifacts).")
    if edge_mag > 25:
        reasons.append("Edge halos/abrupt transitions around facial regions suggest blending or face swap seams.")
    if focus_ratio > 0.25:
        reasons.append("Model attention peaks around eyes/mouth where lip‑sync or blink artifacts commonly appear.")
    if not reasons:
        reasons.append("Subtle inconsistencies in texture and lighting highlighted by the attention map.")
    return reasons[:3]
# ================== end explain_utils.py ==================
