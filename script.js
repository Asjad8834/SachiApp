// script.js
// SACHI v1.1 — TensorFlow YAMNet + local custom training (embedding + cosine-sim) demo

/* ---------- Selectors ---------- */
const listenBtn = document.getElementById('listenBtn');
const listenStatus = document.getElementById('listenStatus');
const micGround = document.getElementById('micGround');
const freqCanvas = document.getElementById('freqCanvas');
const levelText = document.getElementById('levelText');
const labelNameInput = document.getElementById('labelName');
const fileInput = document.getElementById('fileInput');
const uploadTrainBtn = document.getElementById('uploadTrainBtn');
const recordSampleBtn = document.getElementById('recordSampleBtn');
const trainBtn = document.getElementById('trainBtn');
const trainedList = document.getElementById('trainedList');
const logList = document.getElementById('logList');
const exportModelBtn = document.getElementById('exportModelBtn');
const importModelBtn = document.getElementById('importModelBtn');
const importFile = document.getElementById('importFile');
const toggleLoggingBtn = document.getElementById('toggleLoggingBtn');
const clearLogsBtn = document.getElementById('clearLogsBtn');
const modelSummary = document.getElementById('modelSummary');
const storageInfo = document.getElementById('storageInfo');
const thresholdInput = document.getElementById('sensitivity');
const thresholdLabel = document.getElementById('thresholdLabel');
const downloadLogsBtn = document.getElementById('downloadLogsBtn');
const resetModelBtn = document.getElementById('resetModelBtn');

/* ---------- Constants / State ---------- */
const MODEL_KEY = 'sachi_model_v1';
const LOGS_KEY = 'sachi_logs_v1';

let yamnetModel = null;
let audioContext = null;
let micStream = null;
let analyser = null;
let freqDataArray = null;
let rafId = null;
let listening = false;
let lastCapturedVector = null; // for training
let logs = [];
let loggingPaused = false;
let model = { labels: {} }; // custom label -> {vector: Float32Array, examples: number}
const freqBinsTarget = 64; // used for simple fallback features

// mapping for YAMNet classes (friendly categories to icons)
const iconMap = {
  'Speech': 'fa-user',
  'Vehicle': 'fa-car',
  'Music': 'fa-music',
  'Dog': 'fa-dog',
  'Bell': 'fa-bell',
  'Alarm': 'fa-bell',
  'Knock': 'fa-door-closed',
  'Siren': 'fa-bullhorn',
  'Other': 'fa-wave-square'
};

// approximate mapping of yamnet labels to friendly categories (short list)
const yamnetFriendly = [
  {pattern: /speech|speech|conversation|talk/ig, label: 'Speech'},
  {pattern: /car|vehicle|automobile|engine/ig, label: 'Vehicle'},
  {pattern: /dog|bark/ig, label: 'Dog'},
  {pattern: /bell|doorbell|chime/ig, label: 'Bell'},
  {pattern: /siren|ambulance|police/ig, label: 'Siren'},
  {pattern: /music|singing|song/ig, label: 'Music'},
];

/* ---------- Utilities ---------- */
function $(sel){ return document.querySelector(sel); }
function nowIso(){ return new Date().toISOString(); }

function cosineSim(a,b){
  let dot=0, na=0, nb=0;
  for(let i=0;i<a.length;i++){
    dot += a[i]*b[i];
    na += a[i]*a[i];
    nb += b[i]*b[i];
  }
  if(na===0 || nb===0) return 0;
  return dot / (Math.sqrt(na)*Math.sqrt(nb));
}

function normalizeVec(v){
  let sum = 0;
  for(let i=0;i<v.length;i++) sum += Math.abs(v[i]);
  if(sum === 0) return v.map(()=>0);
  return v.map(x => x / sum);
}

/* ---------- Persistence ---------- */
function loadModel(){
  try{
    const raw = localStorage.getItem(MODEL_KEY);
    if(raw){
      const parsed = JSON.parse(raw);
      model.labels = {};
      parsed.forEach(e => {
        model.labels[e.label] = { vector: Float32Array.from(e.vector), examples: e.examples || 1 };
      });
    }else model.labels = {};
  }catch(e){ console.warn('loadModel error', e); model.labels = {}; }
  updateUIModel();
}

function saveModel(){
  const arr = [];
  for(const label in model.labels) arr.push({label, vector: Array.from(model.labels[label].vector), examples: model.labels[label].examples});
  localStorage.setItem(MODEL_KEY, JSON.stringify(arr));
  updateUIModel();
}

function loadLogs(){
  try{
    logs = JSON.parse(localStorage.getItem(LOGS_KEY) || '[]');
  }catch(e){ logs = []; }
  renderLogs();
}

function saveLogs(){
  try{ localStorage.setItem(LOGS_KEY, JSON.stringify(logs)); }catch(e){}
}

/* ---------- UI updates ---------- */
function updateUIModel(){
  trainedList.innerHTML = '';
  const keys = Object.keys(model.labels);
  storageInfo.textContent = `Saved: ${keys.length}`;
  modelSummary.textContent = keys.length ? `${keys.length} trained sound(s)` : 'No trained sounds';
  keys.forEach(k=>{
    const e = model.labels[k];
    const div = document.createElement('div');
    div.className = 'trained-item';
    div.innerHTML = `<div>
      <div style="font-weight:700">${k}</div>
      <div class="meta">${e.examples} sample(s)</div>
    </div>
    <div><button class="btn ghost delete-label" data-label="${k}">Delete</button></div>`;
    trainedList.appendChild(div);
  });
}

function renderLogs(){
  logList.innerHTML = '';
  for(let i=logs.length-1;i>=0;i--){
    const item = logs[i];
    const li = document.createElement('div');
    li.className = 'log-item';
    const iconCls = item.icon || 'fa-wave-square';
    li.innerHTML = `<div class="left">
      <div class="badge"><i class="fas ${iconCls}" style="font-size:14px"></i></div>
      <div style="display:flex;flex-direction:column">
        <div style="font-weight:700">${item.label}</div>
        <div class="muted" style="font-size:13px">${new Date(item.when).toLocaleString()} • ${item.direction||'Unknown'} • sim ${item.similarity.toFixed(2)}</div>
      </div>
    </div>
    <div class="muted small-label">${new Date(item.when).toLocaleTimeString()}</div>`;
    logList.appendChild(li);
  }
}

/* ---------- Audio / visualizer ---------- */
const canvas = freqCanvas;
const ctx = canvas.getContext('2d');
function resizeCanvas(){
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.floor(canvas.clientWidth * ratio);
  canvas.height = Math.floor(canvas.clientHeight * ratio);
}
window.addEventListener('resize', resizeCanvas);
resizeCanvas();

function drawVisualizer(){
  if(!analyser) return;
  analyser.getByteFrequencyData(freqDataArray);
  const W = canvas.width;
  const H = canvas.height;
  ctx.clearRect(0,0,W,H);
  const bins = freqDataArray.length;
  const barW = Math.max(1, (W / bins));
  for(let i=0;i<bins;i++){
    const v = freqDataArray[i] / 255;
    const x = Math.floor(i * barW);
    const h = Math.max(1, Math.floor(v * H));
    const g = ctx.createLinearGradient(0,H,0,H-h);
    g.addColorStop(0, 'rgba(47,128,237,0.14)');
    g.addColorStop(1, 'rgba(86,204,242,0.24)');
    ctx.fillStyle = g;
    ctx.fillRect(x, H-h, Math.ceil(barW-1), h);
  }
}

/* ---------- YAMNet load ---------- */
async function loadYamnet(){
  try{
    modelSummary.textContent = 'Loading pretrained model...';
    yamnetModel = await yamnet.load();
    modelSummary.textContent = 'Pretrained model loaded (YAMNet).';
  }catch(e){
    console.error('yamnet load error', e);
    modelSummary.textContent = 'Failed to load YAMNet — falling back to spectrum-only';
    yamnetModel = null;
  }
}

/* ---------- Resample helper (to 16k) ---------- */
function resampleBuffer(buffer, inputSampleRate, targetRate=16000){
  if(inputSampleRate === targetRate) return buffer;
  const ratio = inputSampleRate / targetRate;
  const newLength = Math.round(buffer.length / ratio);
  const out = new Float32Array(newLength);
  let offsetResult = 0;
  let offsetBuffer = 0;
  while(offsetResult < out.length){
    const nextOffsetBuffer = Math.round((offsetResult + 1) * ratio);
    // average samples between offsetBuffer..nextOffsetBuffer
    let accum = 0, count=0;
    for(let i=offsetBuffer;i<nextOffsetBuffer && i<buffer.length;i++){ accum += buffer[i]; count++; }
    out[offsetResult] = (count>0 ? accum / count : 0);
    offsetResult++;
    offsetBuffer = nextOffsetBuffer;
  }
  return out;
}

/* ---------- Direction estimate (stereo heuristic) ---------- */
function estimateDirection(){
  if(!micStream) return 'Unknown';
  try{
    const src = audioContext.createMediaStreamSource(micStream);
    const splitter = audioContext.createChannelSplitter(2);
    const aL = audioContext.createAnalyser();
    const aR = audioContext.createAnalyser();
    aL.fftSize = 256; aR.fftSize = 256;
    src.connect(splitter);
    splitter.connect(aL, 0);
    splitter.connect(aR, 1);
    const arrL = new Uint8Array(aL.frequencyBinCount);
    const arrR = new Uint8Array(aR.frequencyBinCount);
    aL.getByteFrequencyData(arrL);
    aR.getByteFrequencyData(arrR);
    let eL=0,eR=0;
    for(let i=0;i<arrL.length;i++){ eL += arrL[i]; eR += arrR[i]; }
    if(eL > eR * 1.15) return 'Left';
    if(eR > eL * 1.15) return 'Right';
    return 'Center';
  }catch(e){
    return 'Unknown';
  }
}

/* ---------- Classification: YAMNet + custom KNN-like ---------- */
async function classifyBuffer(float32Buffer, sampleRate){
  // Returns an object: {pretrained: {label,score}, custom: {label,score}, friendly}
  let pretrained = null;
  let embedding = null;

  if(yamnetModel){
    try{
      // YAMNet in TFJS expects a mono Float32Array sampled at 16kHz.
      const mono = float32Buffer; // assume mono already
      const resampled = resampleBuffer(mono, sampleRate, 16000);
      // model.predict expects Tensor1D of waveform
      const tensor = tf.tensor1d(resampled);
      // some TFJS yamnet APIs respond with {scores, embeddings}
      const out = await yamnetModel.predict(tensor);
      // model.predict returns {scores, embeddings} or a tensor of scores.
      // Try to read embeddings & scores robustly:
      if(out && out['scores'] && out['embeddings']){
        // averaged across frames
        const scores = out['scores']; // tf.Tensor shape [numFrames, numClasses] or array
        const embeddings = out['embeddings'];
        // compute mean along frames for embeddings and scores
        const embTensor = embeddings.mean(0);
        const scTensor = scores.mean(0);
        const embArr = await embTensor.array();
        const scArr = await scTensor.array();
        embedding = embArr;
        // top class:
        let maxIdx = 0;
        for(let i=1;i<scArr.length;i++) if(scArr[i] > scArr[maxIdx]) maxIdx = i;
        const className = yamnetModel.classNames ? yamnetModel.classNames[maxIdx] : `class_${maxIdx}`;
        pretrained = {label: className, score: scArr[maxIdx]};
        embTensor.dispose(); scTensor.dispose();
      }else if(Array.isArray(out) || out instanceof tf.Tensor){
        // fallback: try treating out as scores tensor
        let scArr = null;
        if(out instanceof tf.Tensor){
          scArr = await out.mean(0).array();
        }else if(Array.isArray(out)){
          scArr = out;
        }
        let maxIdx = 0;
        for(let i=1;i<scArr.length;i++) if(scArr[i] > scArr[maxIdx]) maxIdx = i;
        const className = yamnetModel.classNames ? yamnetModel.classNames[maxIdx] : `class_${maxIdx}`;
        pretrained = {label: className, score: scArr[maxIdx]};
      }else{
        // unexpected
      }
      tensor.dispose();
    }catch(err){
      console.warn('YAMNet predict failed', err);
    }
  }

  // Compute embedding for custom matching if possible using yamnetModel (or use simple FFT fallback)
  if(yamnetModel && !embedding){
    // try to call embed/infer API if available
    try{
      const resampled = resampleBuffer(float32Buffer, sampleRate, 16000);
      const t = tf.tensor1d(resampled);
      if(typeof yamnetModel.embed === 'function'){
        const emb = await yamnetModel.embed(t); // if exists
        embedding = Array.from(await emb.mean(0).array());
        emb.dispose();
      }
      t.dispose();
    }catch(e){ console.warn('embedding fallback failed', e); }
  }

  // fallback: simple frequency vector feature when YAMNet not available
  if(!embedding){
    const fftVec = computeSimpleSpectrum(float32Buffer, sampleRate, freqBinsTarget);
    embedding = fftVec;
  }

  // compare to custom model labels
  let bestCustom = {label: null, score: 0};
  for(const lbl in model.labels){
    const vec = Array.from(model.labels[lbl].vector);
    const sc = cosineSim(embedding, vec);
    if(sc > bestCustom.score){ bestCustom = {label: lbl, score: sc}; }
  }

  // friendly mapping for pretrained
  let friendly = null;
  if(pretrained && pretrained.label){
    const cname = pretrained.label;
    for(const m of yamnetFriendly){
      if(m.pattern.test(cname)){ friendly = m.label; break; }
    }
    if(!friendly) friendly = 'Other';
  }

  return { pretrained, custom: bestCustom, friendly, embedding };
}

/* ---------- simple FFT-based spectrum (fallback) ---------- */
function computeSimpleSpectrum(float32Buffer, sampleRate, bins=64){
  // compute absolute-windowed FFT via builtin offline approach (coarse)
  // We'll downsample & produce coarse band-energy vector
  const res = resampleBuffer(float32Buffer, sampleRate, 8000);
  const window = 1024;
  const samples = res.subarray(0, Math.min(res.length, window));
  const mags = new Float32Array(bins).fill(0);
  for(let i=0;i<samples.length;i++){
    const v = Math.abs(samples[i]);
    const idx = Math.floor((i / samples.length) * bins);
    mags[idx] += v;
  }
  const norm = normalizeVec(Array.from(mags));
  return norm;
}

/* ---------- capture & training helpers ---------- */
async function captureAudioSampleFromURL(url, captureMs=1400){
  ensureAudioCtx();
  setupAnalyser();
  return new Promise((resolve, reject)=>{
    const audio = new Audio();
    audio.src = url;
    audio.crossOrigin = 'anonymous';
    audio.muted = true;
    const srcNode = audioContext.createMediaElementSource(audio);
    srcNode.connect(analyser);
    analyser.connect(audioContext.destination);
    audio.play().catch(e=>console.warn(e));
    const collects = [];
    const intMs = 80;
    const loops = Math.ceil(captureMs / intMs);
    let done = 0;
    const iv = setInterval(()=>{
      const f = new Float32Array(analyser.fftSize);
      analyser.getFloatTimeDomainData(f);
      collects.push(Float32Array.from(f));
      done++;
      if(done >= loops){
        clearInterval(iv);
        // average
        const sum = new Float32Array(collects[0].length);
        for(const c of collects) for(let i=0;i<c.length;i++) sum[i] += c[i];
        for(let i=0;i<sum.length;i++) sum[i] /= collects.length;
        // get sample as Float32Array
        const out = sum;
        audio.pause();
        try{ srcNode.disconnect(); }catch(e){}
        resolve({samples: out, rate: audioContext.sampleRate});
      }
    }, intMs);
    setTimeout(()=>{ if(!done){ clearInterval(iv); reject('timeout'); } }, captureMs + 1000);
  });
}

recordSampleBtn.addEventListener('click', async ()=>{
  alert('Recording ~1.8s. Make the sound now.');
  try{
    const stream = await navigator.mediaDevices.getUserMedia({audio:true});
    const rec = new MediaRecorder(stream);
    const chunks = [];
    rec.ondataavailable = e => chunks.push(e.data);
    rec.start();
    await new Promise(r => setTimeout(r, 1800));
    rec.stop();
    await new Promise(r => rec.onstop = r);
    const blob = new Blob(chunks, {type:'audio/webm'});
    const url = URL.createObjectURL(blob);
    const captured = await captureAudioSampleFromURL(url, 1600);
    lastCapturedVector = {samples: captured.samples, rate: captured.rate};
    stream.getTracks().forEach(t=>t.stop());
    alert('Captured sample ready. Enter a label and click Save/Train.');
  }catch(e){
    console.error(e);
    alert('Recording failed or microphone access denied.');
  }
});

uploadTrainBtn.addEventListener('click', async ()=>{
  if(!fileInput.files || fileInput.files.length===0){ alert('Choose an audio file first'); return; }
  const f = fileInput.files[0];
  const url = URL.createObjectURL(f);
  try{
    const cap = await captureAudioSampleFromURL(url, 1500);
    lastCapturedVector = {samples: cap.samples, rate: cap.rate};
    alert('File sample captured. Enter label and click Save/Train.');
  }catch(e){ console.error(e); alert('Failed to capture audio from file.'); }
});

trainBtn.addEventListener('click', async ()=>{
  const label = labelNameInput.value.trim();
  if(!label){ alert('Enter a label name'); return; }
  if(!lastCapturedVector){ alert('No captured sample — record or upload first.'); return; }
  // get embedding (prefer yamnet) or fallback to simple spectrum
  let embedding = null;
  try{
    const arr = lastCapturedVector.samples;
    const rate = lastCapturedVector.rate;
    const c = await classifyBuffer(arr, rate);
    embedding = c.embedding;
  }catch(e){ console.warn('train classify error', e); embedding = computeSimpleSpectrum(lastCapturedVector.samples, lastCapturedVector.rate, freqBinsTarget); }

  if(model.labels[label]){
    const entry = model.labels[label];
    const old = Array.from(entry.vector);
    const combined = new Float32Array(old.length);
    for(let i=0;i<old.length;i++) combined[i] = (old[i] * entry.examples + embedding[i]) / (entry.examples + 1);
    model.labels[label] = { vector: Float32Array.from(normalizeVec(Array.from(combined))), examples: entry.examples + 1 };
  }else{
    model.labels[label] = { vector: Float32Array.from(normalizeVec(embedding)), examples: 1 };
  }
  lastCapturedVector = null;
  labelNameInput.value = '';
  saveModel();
  alert(`Trained "${label}".`);
});

/* delete trained label */
trainedList.addEventListener('click', (ev)=>{
  const btn = ev.target.closest('button.delete-label');
  if(!btn) return;
  const lbl = btn.dataset.label;
  if(confirm(`Delete "${lbl}"?`)){ delete model.labels[lbl]; saveModel(); }
});

/* export/import */
exportModelBtn.addEventListener('click', ()=>{
  const arr = [];
  for(const k in model.labels) arr.push({label:k, vector:Array.from(model.labels[k].vector), examples:model.labels[k].examples});
  const blob = new Blob([JSON.stringify(arr,null,2)], {type:'application/json'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a'); a.href = url; a.download = 'sachi-model.json'; a.click();
  URL.revokeObjectURL(url);
});
importModelBtn.addEventListener('click', ()=> importFile.click());
importFile.addEventListener('change', (ev)=>{
  const f = ev.target.files[0]; if(!f) return;
  const reader = new FileReader();
  reader.onload = ()=>{
    try{
      const parsed = JSON.parse(reader.result);
      parsed.forEach(e => model.labels[e.label] = { vector: Float32Array.from(e.vector), examples: e.examples || 1 });
      saveModel();
      alert('Imported model.');
    }catch(e){ alert('Invalid file'); }
  };
  reader.readAsText(f);
});

/* logs */
toggleLoggingBtn.addEventListener('click', ()=>{
  loggingPaused = !loggingPaused;
  toggleLoggingBtn.textContent = loggingPaused ? 'Resume Logs' : 'Pause Logs';
});
clearLogsBtn.addEventListener('click', ()=>{
  if(confirm('Clear all logs?')){ logs=[]; saveLogs(); renderLogs(); }
});
downloadLogsBtn.addEventListener('click', ()=>{
  const blob = new Blob([JSON.stringify(logs,null,2)], {type:'application/json'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a'); a.href = url; a.download = 'sachi-logs.json'; a.click();
  URL.revokeObjectURL(url);
});
resetModelBtn.addEventListener('click', ()=>{
  if(confirm('Reset model and trained sounds?')){
    model.labels = {}; saveModel(); alert('Reset complete.');
  }
});

/* ---------- Start / stop listening ---------- */
listenBtn.addEventListener('click', ()=>{
  if(!listening) startListening(); else stopListening();
});

async function ensureAudioCtx(){
  if(!audioContext) audioContext = new (window.AudioContext || window.webkitAudioContext)();
}

function setupAnalyser(){
  if(!analyser){
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 256;
    freqDataArray = new Uint8Array(analyser.frequencyBinCount);
  }
}

async function startListening(){
  await ensureAudioCtx();
  if(!yamnetModel) await loadYamnet();

  try{
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    micStream = stream;
    const src = audioContext.createMediaStreamSource(stream);
    setupAnalyser();
    src.connect(analyser);
    listenBtn.classList.add('active');
    listenStatus.textContent = 'Listening';
    micGround.textContent = 'Microphone active';
    listening = true;
    document.querySelector('.pulse').style.opacity = '1';
    updateLoop();
  }catch(e){
    console.error(e);
    alert('Microphone access required.');
  }
}

function stopListening(){
  listenBtn.classList.remove('active');
  listenStatus.textContent = 'Idle';
  micGround.textContent = 'Microphone not active';
  if(micStream) micStream.getTracks().forEach(t=>t.stop());
  micStream = null;
  if(rafId) cancelAnimationFrame(rafId);
  listening = false;
  document.querySelector('.pulse').style.opacity = '0';
}

/* classification loop */
let lastClassifiedAt = 0;
const CLASSIFY_INTERVAL_MS = 900;

async function updateLoop(){
  rafId = requestAnimationFrame(updateLoop);
  drawVisualizer();
  // level
  if(analyser && freqDataArray){
    analyser.getByteFrequencyData(freqDataArray);
    let energy = 0; for(let i=0;i<freqDataArray.length;i++) energy += freqDataArray[i];
    levelText.textContent = (energy).toFixed(0) + ' (level)';
  }

  const now = performance.now();
  if(now - lastClassifiedAt < CLASSIFY_INTERVAL_MS) return;
  lastClassifiedAt = now;

  // capture short time-domain sample
  if(!audioContext || !analyser) return;
  const bufferSize = 2048;
  const temp = new Float32Array(bufferSize);
  analyser.getFloatTimeDomainData(temp);
  // convert Float32Array samples to Float32Array (they already are)
  const sampleRate = audioContext.sampleRate;
  // classify
  try{
    const res = await classifyBuffer(temp, sampleRate);
    // choose final label: prefer custom if above threshold, else pretrained
    const threshold = parseFloat(thresholdInput.value);
    let finalLabel = null;
    let finalScore = 0;
    let icon = 'fa-wave-square';
    if(res.custom && res.custom.label && res.custom.score >= threshold){
      finalLabel = res.custom.label;
      finalScore = res.custom.score;
      icon = (modelIconForLabel(finalLabel) || 'fa-wave-square');
    }else if(res.pretrained && res.pretrained.label){
      // map friendly label if possible
      const friendly = res.friendly || res.pretrained.label;
      finalLabel = friendly;
      finalScore = res.pretrained.score || 0;
      // icon map
      icon = iconForFriendly(finalLabel);
    }
    if(finalLabel){
      const direction = estimateDirection();
      const entry = { when: nowIso(), label: finalLabel, similarity: finalScore, direction, icon };
      // push log if not paused and avoid duplicate rapid identical entries
      const last = logs.length ? logs[logs.length-1] : null;
      if(!loggingPaused){
        if(!last || last.label !== entry.label || (new Date(entry.when) - new Date(last.when) > 3000)){
          logs.push(entry);
          saveLogs();
          renderLogs();
          flashDetected();
        }
      }
    }
  }catch(e){
    console.warn('classify error', e);
  }
}

function flashDetected(){
  listenBtn.animate([
    { transform: 'translateY(0) scale(1)' },
    { transform: 'translateY(-6px) scale(1.03)' },
    { transform: 'translateY(0) scale(1)' }
  ], { duration: 420, easing: 'ease-out' });
}

/* ---------- Helpers: icon selection ---------- */
function iconForFriendly(label){
  if(!label) return iconMap['Other'];
  if(iconMap[label]) return iconMap[label];
  const l = label.toLowerCase();
  if(l.includes('speech') || l.includes('talk')) return iconMap.Speech;
  if(l.includes('car') || l.includes('vehicle') || l.includes('engine')) return iconMap.Vehicle;
  if(l.includes('dog') || l.includes('bark')) return iconMap.Dog;
  if(l.includes('bell') || l.includes('doorbell') || l.includes('chime')) return iconMap.Bell;
  if(l.includes('siren')) return iconMap.Siren;
  if(l.includes('music') || l.includes('sing')) return iconMap.Music;
  return iconMap.Other;
}
function modelIconForLabel(label){
  // let user-provided label match some icons heuristically
  return iconForFriendly(label);
}

/* ---------- startup ---------- */
thresholdLabel.textContent = thresholdInput.value;
thresholdInput.addEventListener('input', ()=> thresholdLabel.textContent = thresholdInput.value);

async function init(){
  loadModel();
  loadLogs();
  await loadYamnet();
  modelSummary.textContent = 'Ready';
}
init();
