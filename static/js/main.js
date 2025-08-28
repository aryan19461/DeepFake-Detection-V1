const $ = (id) => document.getElementById(id);

const modelBadge = $("modelBadge");
const dropzone = $("dropzone");
const fileInput = $("fileInput");
const browseBtn = $("browseBtn");
const uploadBtn = $("uploadBtn");
const startCam = $("startCam");
const snapBtn = $("snap");
const video = $("video");
const canvas = $("canvas");
const resultSection = $("resultSection");
const resultBox = $("resultBox");
const loader = $("loader");
const label = $("label");
const prob = $("prob");
const note = $("note");
const overlay = $("overlay");

let selectedFile = null;
let streamActive = false;

fetch("/health").then(r => r.json()).then(d => {
  modelBadge.textContent = d.model_loaded ? "Model: loaded ✅" : "Model: mock mode ⚠️";
  modelBadge.className = d.model_loaded
    ? "inline-block mt-3 text-xs px-3 py-1 rounded-full bg-emerald-800 border border-emerald-600"
    : "inline-block mt-3 text-xs px-3 py-1 rounded-full bg-yellow-800 border border-yellow-600";
});

dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("dragover");
});
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragover");
  const file = e.dataTransfer.files[0];
  if (file) {
    selectedFile = file;
    uploadBtn.disabled = false;
  }
});

browseBtn.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (file) {
    selectedFile = file;
    uploadBtn.disabled = false;
  }
});

uploadBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  showOverlay(true);
  const fd = new FormData();
  fd.append("file", selectedFile);
  try {
    const res = await fetch("/predict", { method: "POST", body: fd });
    const data = await res.json();
    renderResult(data);
  } catch (e) {
    renderError(e);
  } finally {
    showOverlay(false);
  }
});

startCam.addEventListener("click", async () => {
  if (streamActive) return;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    video.srcObject = stream;
    streamActive = true;
    snapBtn.disabled = false;
  } catch (err) {
    alert("Camera access denied or unavailable.");
  }
});

snapBtn.addEventListener("click", async () => {
  if (!streamActive) return;
  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 360;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  const dataUrl = canvas.toDataURL("image/jpeg");

  showOverlay(true);
  try {
    const res = await fetch("/predict_base64", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: dataUrl }),
    });
    const data = await res.json();
    renderResult(data);
  } catch (e) {
    renderError(e);
  } finally {
    showOverlay(false);
  }
});

function renderResult(data){
  resultSection.classList.remove("hidden");
  loader.classList.add("hidden");
  if(data.error){
    label.textContent = "Error";
    prob.textContent = data.error;
    note.textContent = "";
    return;
  }
  label.textContent = `${data.label} (${(data.prob_fake*100).toFixed(1)}% fake probability)`;
  prob.textContent = data.model_loaded ? "Using: Trained Keras model" : "Using: Mock mode (demo heuristic)";
  note.textContent = data.filename ? `File: ${data.filename}` : "";
  anime({ targets: '#resultBox', translateY: [-10, 0], opacity: [0, 1], duration: 600, easing: 'easeOutBack' });
}

function renderError(e){
  resultSection.classList.remove("hidden");
  label.textContent = "Prediction failed";
  prob.textContent = e.toString();
  note.textContent = "";
}

function showOverlay(show){ overlay.classList.toggle("hidden", !show); }
