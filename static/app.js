// static/app.js — live streaming + stable ≥80% TTS

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const btnStart = document.getElementById("btnStart");
const btnSnap = document.getElementById("btnSnap"); // Start/Stop Live Predict
const fileInput = document.getElementById("file");
const statusEl = document.getElementById("status");
const labelEl = document.getElementById("label");
const confEl = document.getElementById("conf");

const voiceToggle = document.getElementById("voiceToggle");
const holdSecondsSel = document.getElementById("holdSeconds");

let chart;
let streaming = false;
let tickHandle = null;

// Send frames at a reasonable size (client center-crops to this aspect)
const SEND_W = 300;
const SEND_H = 360;

const PROB_THRESHOLD = 0.80; // 80%+

// Stability tracking
let currentTop = null;       // current top label under consideration
let stableSince = 0;         // when the label became stable (ms)
let lastAnnounced = { label: null, time: 0 }; // prevent repeat spam

function initChart() {
  const ctx = document.getElementById("chart").getContext("2d");
  chart = new Chart(ctx, {
    type: "bar",
    data: { labels: [], datasets: [{ label: "Probability", data: [] }] },
    options: {
      indexAxis: "y",
      scales: { x: { min: 0, max: 1, ticks: { stepSize: 0.1 } } }
    }
  });
}
initChart();

async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    video.srcObject = stream;
    btnSnap.disabled = false;
    statusEl.textContent = "Status: camera ready";
  } catch (e) {
    statusEl.textContent = "Status: camera error — " + e.message;
    console.error(e);
  }
}

// Center-crop to 5:6 then scale to SEND_W×SEND_H
function grabFrameDataURL(el) {
  const off = document.createElement("canvas");
  off.width = SEND_W;
  off.height = SEND_H;
  const ctx = off.getContext("2d");
  const w = el.videoWidth || el.naturalWidth || el.width;
  const h = el.videoHeight || el.naturalHeight || el.height;
  if (!w || !h) return null;

  const targetAR = SEND_W / SEND_H; // 0.833...
  const srcAR = w / h;
  let sx, sy, sw, sh;
  if (srcAR > targetAR) { // crop width
    sh = h; sw = Math.floor(h * targetAR); sx = Math.floor((w - sw) / 2); sy = 0;
  } else { // crop height
    sw = w; sh = Math.floor(w / targetAR); sx = 0; sy = Math.floor((h - sh) / 2);
  }
  ctx.drawImage(el, sx, sy, sw, sh, 0, 0, SEND_W, SEND_H);
  return off.toDataURL("image/jpeg", 0.8);
}

async function postFrame(dataURL) {
  const res = await fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: dataURL })
  });
  const out = await res.json();
  if (!res.ok) throw new Error(out.error || res.statusText);
  return out;
}

function updateChart(probs) {
  const entries = Object.entries(probs || {}).sort((a, b) => b[1] - a[1]);
  const top = entries.slice(0, Math.min(10, entries.length));
  chart.data.labels = top.map(([k]) => k);
  chart.data.datasets[0].data = top.map(([, v]) => v);
  chart.update();
}

// --- Web Speech ---
let voiceEnabled = false;
voiceToggle?.addEventListener("change", (e) => {
  voiceEnabled = e.target.checked;
  // Some browsers require a user gesture before speech works — prime it
  if (voiceEnabled && "speechSynthesis" in window) {
    window.speechSynthesis.cancel();
    // Optional: play a tiny "enabled" cue
    const u = new SpeechSynthesisUtterance("Voice enabled");
    window.speechSynthesis.speak(u);
  }
});

function speak(text) {
  if (!voiceEnabled || !("speechSynthesis" in window)) return;
  const now = Date.now();

  // avoid repeating same label too often
  if (lastAnnounced.label === text && (now - lastAnnounced.time) < 4000) return;

  const u = new SpeechSynthesisUtterance(text);
  u.lang = "en-US";   // adjust if needed
  u.rate = 1.0;
  u.pitch = 1.0;

  window.speechSynthesis.cancel(); // clear queue so we speak only once per event
  window.speechSynthesis.speak(u);
  lastAnnounced = { label: text, time: now };
}

function maybeAnnounceStable(topLabel, topProb) {
  const holdMs = (parseInt(holdSecondsSel?.value || "3", 10) || 3) * 1000;

  if (topProb >= PROB_THRESHOLD) {
    if (topLabel === currentTop) {
      if (!stableSince) stableSince = Date.now();
      const elapsed = Date.now() - stableSince;
      // Speak once when we've held for the chosen duration
      if (elapsed >= holdMs) {
        speak(topLabel);
        // After speaking, continue tracking, but don't repeat unless label changes or cooldown passes
        stableSince = Date.now(); // reset so we wait another full window before any re-announce
      }
    } else {
      // label changed: start new timer
      currentTop = topLabel;
      stableSince = Date.now();
    }
  } else {
    // below threshold: reset stability
    if (topLabel !== currentTop) currentTop = null;
    stableSince = 0;
  }
}

async function tick() {
  if (!streaming) return;
  const dataURL = grabFrameDataURL(video);
  if (!dataURL) return;

  try {
    const out = await postFrame(dataURL);
    if (out.status === "collecting") {
      statusEl.textContent = `Status: collecting frames ${out.have}/${out.need}`;
      return;
    }
    if (out.status === "ok") {
      statusEl.textContent = "Status: predicted";
      labelEl.textContent = out.label;
      confEl.textContent = `(${(out.confidence * 100).toFixed(1)}%)`;
      updateChart(out.probs);

      // Stability + TTS
      maybeAnnounceStable(out.label, out.confidence);
    }
  } catch (e) {
    statusEl.textContent = "Status: error — " + e.message;
    console.error(e);
  }
}

btnStart.addEventListener("click", startCamera);

// Toggle live prediction
btnSnap.addEventListener("click", async () => {
  streaming = !streaming;
  btnSnap.textContent = streaming ? "Stop Live Predict" : "Start Live Predict";

  // reset stability and server buffer
  currentTop = null;
  stableSince = 0;
  lastAnnounced = { label: null, time: 0 };
  await fetch("/reset", { method: "POST" }).catch(() => {});

  if (streaming) {
    statusEl.textContent = "Status: collecting frames 0/30";
    tickHandle = setInterval(tick, 120); // ~8 FPS
  } else {
    clearInterval(tickHandle);
  }
});

// Optional: single image upload path still supported (fills buffer incrementally)
fileInput.addEventListener("change", async (e) => {
  const file = e.target.files?.[0];
  if (!file) return;
  const img = new Image();
  img.onload = async () => {
    const dataURL = grabFrameDataURL(img);
    try {
      const out = await postFrame(dataURL);
      if (out.status === "collecting") {
        statusEl.textContent = `Status: collecting frames ${out.have}/${out.need}`;
      } else {
        labelEl.textContent = out.label;
        confEl.textContent = `(${(out.confidence * 100).toFixed(1)}%)`;
        updateChart(out.probs);
        maybeAnnounceStable(out.label, out.confidence);
        statusEl.textContent = "Status: predicted";
      }
    } catch (e2) {
      statusEl.textContent = "Status: error — " + e2.message;
    }
  };
  img.src = URL.createObjectURL(file);
});
