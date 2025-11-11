// SACHI v1.2.6 — complete JS (Compass + robust Speech mapping + relaxed gating)
// Changes vs your last file:
// 1) Lowered PRETRAINED_MIN_CONF to 0.08 and added dynamic override for Speech when RMS is high.
// 2) Never silently return if we have a label; we always emit a log when VAD is open.
// 3) Kept your compass + class_map fixes. Nothing else touched.

// ======================== Selectors ========================
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
const modelSelector = document.getElementById('modelSelector');
const debugToggle = document.getElementById('debugToggle');
const debugPanel = document.getElementById('debugPanel');
const debugBody = document.getElementById('debugBody');

// Compass (optional)
const compassCanvas = document.getElementById('compassCanvas');
const dirText = document.getElementById('dirText');
const azText = document.getElementById('azText');

// ======================== Constants / State ========================
const MODEL_KEY = 'sachi_model_v1';
const LOGS_KEY = 'sachi_logs_v1';

const TFHUB_YAMNET_URL = 'https://tfhub.dev/google/tfjs-model/yamnet/classification/1/default/1';
const LOCAL_YAMNET_PATH = './models/yamnet/model.json';
const CLASS_MAP_URL = './models/yamnet/class_map.json';

const MIN_YAMNET_LEN = 15600;
const TARGET_SR = 16000;
const RING_SECONDS = 1.0;
const CLASSIFY_INTERVAL_MS = 900;

// ---- RELAXED GATING (fix for empty logs) ----
const PRETRAINED_MIN_CONF = 0.08;     // was 0.15
const SPEECH_RMS_OVERRIDE = 0.028;    // if RMS >= this and label looks like speech, log anyway

const VAD_RMS_GATE = 0.015;
const SMOOTH_WINDOW = 5;
const FREQ_BINS_TARGET = 64;

let yamnetModel = null;
let CLASS_MAP = null;

let audioContext = null;
let micStream = null;
let analyser = null;
let freqDataArray = null;
let timeDataArray = null;
let rafId = null;
let listening = false;

let lastCapturedVector = null;
let logs = [];
let loggingPaused = false;

let model = { labels: {} };

let ringBuffer = new Float32Array(16000);
let ringWrite = 0;
let ringTotalWritten = 0;

// Stereo compass nodes/state
let splitNode = null, analyserL = null, analyserR = null;
let timeL = null, timeR = null;
let compassAngleTarget = 0;    // -90..+90
let compassAngleSmoothed = 0;  // animated
let lastDirLabel = 'Unknown';
let lastAzDeg = 0;
let lastEnergyRMS = 0;

const iconMap = {
  Speech: 'fa-user',
  Vehicle: 'fa-car',
  Music:  'fa-music',
  Dog:    'fa-dog',
  Bell:   'fa-bell',
  Siren:  'fa-bullhorn',
  Other:  'fa-wave-square'
};
const yamnetFriendly = [
  { pattern: /speech|conversation|talk|narration|whisper|spoken|voice/ig, label: 'Speech' },
  { pattern: /car|vehicle|automobile|engine|horn|traffic/ig, label: 'Vehicle' },
  { pattern: /dog|bark|woof/ig, label: 'Dog' },
  { pattern: /bell|doorbell|chime/ig, label: 'Bell' },
  { pattern: /siren|ambulance|police|fire truck/ig, label: 'Siren' },
  { pattern: /music|sing|song|melody|instrument|piano|guitar|violin|drum/ig, label: 'Music' }
];

// ======================== Utils ========================
function nowIso(){ return new Date().toISOString(); }
function cosineSim(a,b){
  if(!a||!b||a.length!==b.length) return 0;
  let dot=0,na=0,nb=0;
  for(let i=0;i<a.length;i++){ const x=a[i], y=b[i]; dot+=x*y; na+=x*x; nb+=y*y; }
  return (na&&nb) ? dot/(Math.sqrt(na)*Math.sqrt(nb)) : 0;
}
function normalizeVec(v){
  let s=0; for(let i=0;i<v.length;i++) s+=Math.abs(v[i]);
  return s ? v.map(x=>x/s) : v.map(()=>0);
}
function loadScriptDynamic(src){
  return new Promise((resolve,reject)=>{
    const s=document.createElement('script');
    s.src=src; s.async=true;
    s.onload=resolve; s.onerror=()=>reject(new Error('Load failed: '+src));
    document.head.appendChild(s);
  });
}
function friendlyFromRawLabel(raw){
  if(!raw) return null;
  if(/^class_\d+$/i.test(raw)) return null;
  for(const m of yamnetFriendly){ if(m.pattern.test(raw)) return m.label; }
  return null;
}
function iconForFriendly(label){
  if(!label) return iconMap.Other;
  if(iconMap[label]) return iconMap[label];
  const l=label.toLowerCase();
  if(l.includes('speech')||l.includes('talk')||l.includes('voice')) return iconMap.Speech;
  if(l.includes('car')||l.includes('vehicle')||l.includes('engine')||l.includes('horn')) return iconMap.Vehicle;
  if(l.includes('dog')||l.includes('bark')) return iconMap.Dog;
  if(l.includes('bell')||l.includes('doorbell')||l.includes('chime')) return iconMap.Bell;
  if(l.includes('siren')) return iconMap.Siren;
  if(l.includes('music')||l.includes('sing')||l.includes('song')||l.includes('instrument')) return iconMap.Music;
  return iconMap.Other;
}

// ======================== Persistence ========================
function loadModel(){
  try{
    const raw = localStorage.getItem(MODEL_KEY);
    if(raw){
      const parsed = JSON.parse(raw);
      model.labels = {};
      parsed.forEach(e=>{
        model.labels[e.label] = { vector: Float32Array.from(e.vector), examples: e.examples||1 };
      });
    }
  }catch{ model.labels = {}; }
  updateUIModel();
}
function saveModel(){
  const arr = Object.entries(model.labels).map(([label,v])=>({
    label, vector:Array.from(v.vector), examples:v.examples
  }));
  localStorage.setItem(MODEL_KEY, JSON.stringify(arr));
  updateUIModel();
}
function loadLogs(){ try{ logs = JSON.parse(localStorage.getItem(LOGS_KEY)||'[]'); }catch{ logs=[]; } renderLogs(); }
function saveLogs(){ try{ localStorage.setItem(LOGS_KEY, JSON.stringify(logs)); }catch{} }

// ======================== UI ========================
function updateUIModel(){
  trainedList.innerHTML = '';
  const keys = Object.keys(model.labels);
  storageInfo.textContent = `Saved: ${keys.length}`;
  modelSummary.textContent = keys.length ? `${keys.length} trained sound(s)` : 'No trained sounds';
  keys.forEach(k=>{
    const e=model.labels[k];
    const div=document.createElement('div');
    div.className='trained-item';
    div.innerHTML = `<div>
        <div style="font-weight:700">${k}</div>
        <div class="meta">${e.examples} sample(s)</div>
      </div>
      <div><button class="btn ghost delete-label" data-label="${k}">Delete</button></div>`;
    trainedList.appendChild(div);
  });
}
function renderLogs(){
  logList.innerHTML='';
  if(!logs.length){
    const p=document.createElement('div'); p.className='muted'; p.style.padding='12px';
    p.textContent='No detections yet'; logList.appendChild(p); return;
  }
  for(let i=logs.length-1;i>=0;i--){
    const it = logs[i];
    const whenISO = it.when || nowIso();
    const whenLocal = new Date(whenISO).toLocaleString();
    const sim = typeof it.similarity==='number' && !Number.isNaN(it.similarity) ? it.similarity.toFixed(2) : '0.00';
    const conf = typeof it.confidence==='number' && !Number.isNaN(it.confidence) ? it.confidence.toFixed(2) : '—';
    const iconCls = it.icon || 'fa-wave-square';
    const dir = it.direction || 'Unknown';
    const li=document.createElement('div');
    li.className='log-item';
    li.innerHTML = `<div class="left">
        <div class="badge"><i class="fa-solid ${iconCls}" style="font-size:14px"></i></div>
        <div style="display:flex;flex-direction:column">
          <div style="font-weight:700">${it.label || 'Unknown'}</div>
          <div class="muted" style="font-size:13px">${whenLocal} • ${dir} • sim ${sim} • conf ${conf}</div>
        </div>
      </div>
      <div class="muted small-label">${new Date(whenISO).toLocaleTimeString()}</div>`;
    logList.appendChild(li);
  }
}

// ======================== Visualizer ========================
const canvas = freqCanvas;
const ctx = canvas.getContext('2d');
function fixCanvasSize(){
  const ratio = window.devicePixelRatio || 1;
  const cssW = freqCanvas.clientWidth || 520;
  const cssH = (freqCanvas.clientHeight && freqCanvas.clientHeight > 0) ? freqCanvas.clientHeight : 72;
  canvas.width = Math.round(cssW * ratio);
  canvas.height = Math.round(cssH * ratio);
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
}
window.addEventListener('resize', fixCanvasSize);
fixCanvasSize();

function drawVisualizer(){
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0,0,W,H);
  const bg = ctx.createLinearGradient(0,0,0,H);
  bg.addColorStop(0,'rgba(255,255,255,0.03)');
  bg.addColorStop(1,'rgba(255,255,255,0.01)');
  ctx.fillStyle = bg; ctx.fillRect(0,0,W,H);
  ctx.fillStyle = 'rgba(255,255,255,0.08)';
  ctx.fillRect(0, H-2, W, 2);

  if(!analyser || !freqDataArray) return;

  analyser.getByteFrequencyData(freqDataArray);
  const bins = freqDataArray.length;
  const barW = Math.max(1, W / bins);
  for(let i=0;i<bins;i++){
    const v = freqDataArray[i]/255;
    const x = i*barW;
    const h = Math.max(1, v * (H*0.9));
    const alpha = 0.18 + v*0.7;
    ctx.fillStyle = `rgba(86,204,242,${alpha.toFixed(3)})`;
    ctx.fillRect(x, H-h, barW-0.6, h);
  }
}

// ======================== Compass ========================
let compassCtx = null;
if (compassCanvas) {
  const ratio = window.devicePixelRatio || 1;
  const size = 260;
  compassCanvas.width = size * ratio;
  compassCanvas.height = size * ratio;
  compassCtx = compassCanvas.getContext('2d');
  compassCtx.setTransform(ratio, 0, 0, ratio, 0, 0);
}
function drawCompass(angleDeg, energy=0){
  if(!compassCtx) return;
  const ctxC = compassCtx;
  const W = compassCanvas.width / (window.devicePixelRatio||1);
  const H = compassCanvas.height / (window.devicePixelRatio||1);
  const cx = W/2, cy = H/2, r = Math.min(W,H)*0.42;

  ctxC.clearRect(0,0,W,H);

  const grd = ctxC.createRadialGradient(cx, cy, r*0.2, cx, cy, r*1.1);
  grd.addColorStop(0, '#0f1a2c'); grd.addColorStop(1, '#0a1020');
  ctxC.fillStyle = grd; ctxC.beginPath(); ctxC.arc(cx,cy,r*1.05,0,Math.PI*2); ctxC.fill();

  ctxC.lineWidth = 12;
  ctxC.strokeStyle = '#56CCF2'; ctxC.beginPath(); ctxC.arc(cx,cy,r,Math.PI*0.65,Math.PI*1.35); ctxC.stroke();
  ctxC.strokeStyle = '#2F80ED'; ctxC.beginPath(); ctxC.arc(cx,cy,r,Math.PI*1.35,Math.PI*0.65,true); ctxC.stroke();

  ctxC.save(); ctxC.translate(cx,cy);
  for(let i=-90;i<=90;i+=10){
    const a=(i-90)*Math.PI/180, len=(i%30===0)?12:6;
    ctxC.strokeStyle='rgba(255,255,255,0.25)'; ctxC.lineWidth=2;
    const x1=Math.cos(a)*(r-6), y1=Math.sin(a)*(r-6);
    const x2=Math.cos(a)*(r-6-len), y2=Math.sin(a)*(r-6-len);
    ctxC.beginPath(); ctxC.moveTo(x1,y1); ctxC.lineTo(x2,y2); ctxC.stroke();
  }
  ctxC.restore();

  const aRad = (angleDeg - 90) * Math.PI/180;
  const needleLen = r*0.9;
  ctxC.save(); ctxC.translate(cx,cy); ctxC.rotate(aRad);
  ctxC.strokeStyle='rgba(0,0,0,0.45)'; ctxC.lineWidth=8;
  ctxC.beginPath(); ctxC.moveTo(0,0); ctxC.lineTo(needleLen,0); ctxC.stroke();
  ctxC.strokeStyle = energy>0.02 ? '#7CF9A5' : '#9aa7b2';
  ctxC.lineWidth=6; ctxC.beginPath(); ctxC.moveTo(0,0); ctxC.lineTo(needleLen,0); ctxC.stroke();
  ctxC.fillStyle='#e6eef6'; ctxC.beginPath(); ctxC.arc(0,0,6,0,Math.PI*2); ctxC.fill();
  ctxC.restore();
}
function animateCompass(){
  const alpha = 0.15;
  compassAngleSmoothed += alpha*(compassAngleTarget - compassAngleSmoothed);
  drawCompass(compassAngleSmoothed, lastEnergyRMS);
  requestAnimationFrame(animateCompass);
}

// ======================== Model loading ========================
async function loadYamnet(){
  modelSummary.textContent = 'Loading pretrained model...';

  try{
    yamnetModel = await tf.loadGraphModel(LOCAL_YAMNET_PATH);
    yamnetModel.__isLocal = true;
    modelSummary.textContent = 'Local YAMNet TFJS GraphModel loaded.';
    return;
  }catch(e){ console.warn('[YAMNet] local load failed, fallback...', e); }

  try{
    if(typeof yamnet === 'undefined' || !yamnet){
      await loadScriptDynamic('https://cdn.jsdelivr.net/npm/@tensorflow-models/yamnet@1.0.2/dist/yamnet.min.js');
      await new Promise(r=>setTimeout(r,100));
    }
    if(yamnet && typeof yamnet.load==='function'){
      yamnetModel = await yamnet.load();
      modelSummary.textContent = 'Pretrained model loaded (YAMNet)';
      return;
    }
  }catch(e){ console.warn('[YAMNet] npm load failed', e); }

  try{
    yamnetModel = await tf.loadGraphModel(TFHUB_YAMNET_URL, { fromTFHub:true });
    yamnetModel.__isTFHubGraphModel = true;
    modelSummary.textContent = 'Pretrained model loaded (TFHub)';
  }catch(e){
    console.error('[YAMNet] TFHub load failed', e);
    modelSummary.textContent = 'Failed to load YAMNet — using spectrum fallback';
    yamnetModel = null;
  }
}

// Robust class-map loader (accepts multiple shapes)
async function loadClassMap(){
  try{
    const res = await fetch(CLASS_MAP_URL, { cache:'no-store' });
    if(!res.ok) throw new Error('class_map.json not found');
    const data = await res.json();

    const tryExtract = (d)=>{
      if(Array.isArray(d) && typeof d[0]==='string') return d.slice();
      if(d && Array.isArray(d.class_names)) return d.class_names.slice();
      if(d && Array.isArray(d.classes) && typeof d.classes[0]==='string') return d.classes.slice();
      if(d && Array.isArray(d.classes) && typeof d.classes[0]==='object'){
        return d.classes.map(x=> x.display_name || x.name || x.label || '');
      }
      return null;
    };

    let arr = tryExtract(data);
    if(!arr) throw new Error('Unsupported class_map format');
    arr = arr.map((x,idx)=> (x && String(x).trim().length) ? String(x) : `class_${idx}`);
    CLASS_MAP = arr;
    console.info('[YAMNet] class_map loaded:', CLASS_MAP.length);
  }catch(e){
    CLASS_MAP = null;
    console.warn('No/invalid class_map.json — labels may show as class_###', e);
  }
}

// Coerce to Speech/friendly where possible
function sanitizeLabel(rawLabel, top5){
  const looksSpeech = (s)=> /speech|talk|narration|whisper|spoken|voice/i.test(s||'');
  if(looksSpeech(rawLabel)) return 'Speech';
  if(Array.isArray(top5) && top5.some(t=> looksSpeech(t.label))) return 'Speech';
  const fr = friendlyFromRawLabel(rawLabel);
  if(fr) return fr;
  if(/^class_\d+$/i.test(rawLabel) && Array.isArray(top5)){
    for(const t of top5){
      const f = friendlyFromRawLabel(t.label);
      if(f) return f;
    }
  }
  return rawLabel || 'Other';
}

// ======================== Audio helpers ========================
async function ensureAudioCtx(){
  if(!audioContext) audioContext = new (window.AudioContext || window.webkitAudioContext)();
}
function setupAnalyser(){
  if(!analyser){
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 2048;
    analyser.smoothingTimeConstant = 0.8;
    analyser.minDecibels = -90;
    analyser.maxDecibels = -10;
    freqDataArray = new Uint8Array(analyser.frequencyBinCount);
    timeDataArray = new Float32Array(analyser.fftSize);
  }
}
function resampleBuffer(buffer, inputSR, targetSR=TARGET_SR){
  if(inputSR===targetSR) return buffer;
  const ratio = inputSR/targetSR;
  const newLen = Math.max(MIN_YAMNET_LEN, Math.round(buffer.length/ratio));
  const out = new Float32Array(newLen);
  let o=0, i0=0;
  while(o<newLen){
    const i1 = Math.round((o+1)*ratio);
    let acc=0, cnt=0;
    for(let i=i0;i<i1 && i<buffer.length;i++){ acc+=buffer[i]; cnt++; }
    out[o] = cnt ? acc/cnt : 0;
    o++; i0=i1;
  }
  return out;
}
function padToMinLen(arr, minLen=MIN_YAMNET_LEN){
  if(arr.length>=minLen) return arr;
  const out = new Float32Array(minLen); out.set(arr); return out;
}
function estimateDirection(){ return lastDirLabel || 'Unknown'; }
function computeSimpleSpectrum(float32Buffer, sampleRate, bins=FREQ_BINS_TARGET){
  const res = resampleBuffer(float32Buffer, sampleRate, 8000);
  const window = 1024;
  const samples = res.subarray(0, Math.min(res.length, window));
  const mags = new Float32Array(bins).fill(0);
  for(let i=0;i<samples.length;i++){
    const v=Math.abs(samples[i]);
    const idx=Math.floor((i/samples.length)*bins);
    mags[idx]+=v;
  }
  return normalizeVec(Array.from(mags));
}

// ======================== Classification + Top-5 ========================
const smoothQueue = [];
function pushAndVote(pred){
  smoothQueue.push(pred);
  while(smoothQueue.length > SMOOTH_WINDOW) smoothQueue.shift();
  const counts=new Map(), sums=new Map();
  for(const p of smoothQueue){
    const k=p.label||'Unknown';
    counts.set(k,(counts.get(k)||0)+1);
    sums.set(k,(sums.get(k)||0)+(p.score||0));
  }
  let winner=null, maxC=-1;
  for(const [k,c] of counts.entries()){ if(c>maxC){maxC=c;winner=k;} }
  const avg = (sums.get(winner)||0)/(counts.get(winner)||1);
  return { label:winner, score:avg, friendly: pred.friendly || friendlyFromRawLabel(winner) };
}
function renderTop5(top5){
  if(!debugBody) return;
  debugBody.innerHTML='';
  if(!Array.isArray(top5)||!top5.length){
    const tr=document.createElement('tr'); tr.innerHTML='<td>—</td><td>—</td><td>—</td>';
    debugBody.appendChild(tr); return;
  }
  top5.forEach((t,i)=>{
    const tr=document.createElement('tr');
    tr.innerHTML=`<td>#${i+1}</td><td>${t.label}</td><td>${(t.score||0).toFixed(3)}</td>`;
    debugBody.appendChild(tr);
  });
}

async function classifyBuffer(float32Buffer, sampleRate){
  let pretrained=null, embedding=null, top5=null;

  let waveform = resampleBuffer(float32Buffer, sampleRate, TARGET_SR);
  waveform = padToMinLen(waveform, MIN_YAMNET_LEN);

  if(yamnetModel){
    try{
      if(yamnetModel.__isLocal || yamnetModel.__isTFHubGraphModel){
        const t = tf.tensor1d(waveform);
        let out;
        const inputName = (yamnetModel.inputs && yamnetModel.inputs[0] && yamnetModel.inputs[0].name) || 'waveform';
        try{ out = yamnetModel.execute({ [inputName]: t }); }catch{ out = yamnetModel.predict(t); }

        let scoresArr=null;
        if(out instanceof tf.Tensor){
          scoresArr = await out.data();
          out.dispose && out.dispose();
        }else if(out && out['scores']){
          const sc = out['scores'];
          scoresArr = await (sc.mean ? sc.mean(0).data() : sc.data());
        }else if(Array.isArray(out)){
          scoresArr = await out[0].data();
        }

        if(scoresArr){
          const idScores = scoresArr.map((v,idx)=>({idx, v})).sort((a,b)=> b.v-a.v);
          const max = idScores[0];
          const idxToName = (i)=> (CLASS_MAP && CLASS_MAP[i]) ? CLASS_MAP[i] : `class_${i}`;
          const maxName = idxToName(max.idx);
          top5 = idScores.slice(0,5).map(e=>({ label: idxToName(e.idx), score: e.v }));
          const clean = sanitizeLabel(maxName, top5);
          pretrained = { label: clean, score: scoresArr[max.idx], rawIndex: max.idx };
          embedding = Array.from(scoresArr);
        }
        t.dispose && t.dispose();
      }else if(typeof yamnetModel.classify==='function'){
        const preds = await yamnetModel.classify(waveform);
        if(Array.isArray(preds) && preds.length){
          const sorted = preds.slice().sort((a,b)=> b.score-a.score);
          const top = sorted[0];
          const clean = sanitizeLabel(top.className || 'Other', sorted);
          pretrained = { label: clean, score: +top.score || 0 };
          top5 = sorted.slice(0,5).map(p=>({label:sanitizeLabel(p.className,preds), score:+p.score||0}));
        }
        if(typeof yamnetModel.embed==='function'){
          const embT = await yamnetModel.embed(waveform);
          const embArr = embT.rank===2 ? await embT.mean(0).array() : await embT.array();
          embedding = Array.from(embArr);
          embT.dispose();
        }
      }
    }catch(err){ console.warn('YAMNet classification failed:', err); }
  }

  if(!embedding) embedding = computeSimpleSpectrum(float32Buffer, sampleRate, FREQ_BINS_TARGET);

  // Custom cosine-sim
  let bestCustom = { label:null, score:0 };
  for(const lbl in model.labels){
    const sc = cosineSim(embedding, Array.from(model.labels[lbl].vector));
    if(sc > bestCustom.score) bestCustom = { label: lbl, score: sc };
  }

  let friendly = null;
  if(pretrained && pretrained.label) friendly = friendlyFromRawLabel(pretrained.label);

  return { pretrained, custom: bestCustom, friendly, embedding, top5 };
}

// ======================== Capture & Train ========================
async function captureAudioSampleFromURL(url, captureMs=1400){
  await ensureAudioCtx(); setupAnalyser();
  return new Promise((resolve, reject)=>{
    const audio=new Audio(); audio.src=url; audio.crossOrigin='anonymous'; audio.muted=true;
    let srcNode=null; try{ srcNode=audioContext.createMediaElementSource(audio);}catch{}
    if(srcNode) srcNode.connect(analyser);
    audio.play().catch(()=>{});
    const collects=[]; const intMs=80; const loops=Math.ceil(captureMs/intMs); let done=0;
    const iv=setInterval(()=>{
      const f=new Float32Array(analyser.fftSize);
      analyser.getFloatTimeDomainData(f);
      collects.push(Float32Array.from(f)); done++;
      if(done>=loops){
        clearInterval(iv);
        const sum=new Float32Array(collects[0].length);
        for(const c of collects) for(let i=0;i<c.length;i++) sum[i]+=c[i];
        for(let i=0;i<sum.length;i++) sum[i]/=collects.length;
        try{ audio.pause(); }catch{}; try{ srcNode && srcNode.disconnect(); }catch{}
        resolve({samples: sum, rate: audioContext.sampleRate});
      }
    }, intMs);
    setTimeout(()=>{ if(!done){ clearInterval(iv); try{audio.pause();}catch{}; reject('timeout'); } }, captureMs+1000);
  });
}

recordSampleBtn.addEventListener('click', async ()=>{
  alert('Recording ~1.8s. Make the sound now.');
  try{
    const stream=await navigator.mediaDevices.getUserMedia({audio:true});
    const rec=new MediaRecorder(stream); const chunks=[];
    rec.ondataavailable=e=>chunks.push(e.data);
    rec.start(); await new Promise(r=>setTimeout(r,1800)); rec.stop();
    await new Promise(r=>rec.onstop=r);
    const url=URL.createObjectURL(new Blob(chunks,{type:'audio/webm'}));
    const cap=await captureAudioSampleFromURL(url,1600);
    lastCapturedVector={samples:cap.samples, rate:cap.rate};
    stream.getTracks().forEach(t=>t.stop());
    alert('Captured sample ready. Enter a label and click Save/Train.');
  }catch(e){ console.error(e); alert('Recording failed or microphone access denied.'); }
});

uploadTrainBtn.addEventListener('click', async ()=>{
  if(!fileInput.files?.length){ alert('Choose an audio file first'); return; }
  const url=URL.createObjectURL(fileInput.files[0]);
  try{
    const cap=await captureAudioSampleFromURL(url,1500);
    lastCapturedVector={samples:cap.samples, rate:cap.rate};
    alert('File sample captured. Enter label and click Save/Train.');
  }catch(e){ console.error(e); alert('Failed to capture audio from file.'); }
});

trainBtn.addEventListener('click', async ()=>{
  const label=labelNameInput.value.trim();
  if(!label){ alert('Enter a label name'); return; }
  if(!lastCapturedVector){ alert('No captured sample — record or upload first.'); return; }
  let embedding=null;
  try{
    const c=await classifyBuffer(lastCapturedVector.samples, lastCapturedVector.rate);
    embedding=c.embedding;
  }catch{
    embedding = computeSimpleSpectrum(lastCapturedVector.samples, lastCapturedVector.rate, FREQ_BINS_TARGET);
  }
  if(!embedding?.length){ alert('Embedding extraction failed.'); return; }

  if(model.labels[label]){
    const e=model.labels[label]; const old=Array.from(e.vector);
    const combined=new Float32Array(old.length);
    for(let i=0;i<old.length;i++) combined[i]=(old[i]*e.examples + embedding[i])/(e.examples+1);
    model.labels[label]={ vector: Float32Array.from(normalizeVec(Array.from(combined))), examples:e.examples+1 };
  }else{
    model.labels[label]={ vector: Float32Array.from(normalizeVec(embedding)), examples:1 };
  }
  lastCapturedVector=null; labelNameInput.value=''; saveModel(); alert(`Trained "${label}".`);
});

trainedList.addEventListener('click', (ev)=>{
  const btn=ev.target.closest('button.delete-label'); if(!btn) return;
  const lbl=btn.dataset.label;
  if(confirm(`Delete "${lbl}"?`)){ delete model.labels[lbl]; saveModel(); }
});

exportModelBtn.addEventListener('click', ()=>{
  const arr = Object.entries(model.labels).map(([k,v])=>({label:k, vector:Array.from(v.vector), examples:v.examples}));
  const blob=new Blob([JSON.stringify(arr,null,2)],{type:'application/json'});
  const url=URL.createObjectURL(blob); const a=document.createElement('a');
  a.href=url; a.download='sachi-model.json'; a.click(); URL.revokeObjectURL(url);
});
importModelBtn.addEventListener('click', ()=> importFile.click());
importFile.addEventListener('change', (ev)=>{
  const f=ev.target.files[0]; if(!f) return;
  const fr=new FileReader();
  fr.onload=()=>{
    try{
      const parsed=JSON.parse(fr.result);
      model.labels={};
      parsed.forEach(e=> model.labels[e.label]={ vector: Float32Array.from(e.vector), examples:e.examples||1 });
      saveModel(); alert('Imported model.');
    }catch{ alert('Invalid file'); }
  };
  fr.readAsText(f);
});

// logging controls
toggleLoggingBtn.addEventListener('click', ()=>{
  loggingPaused=!loggingPaused;
  toggleLoggingBtn.textContent = loggingPaused ? 'Resume Logs' : 'Pause Logs';
});
clearLogsBtn.addEventListener('click', ()=>{
  if(confirm('Clear all logs?')){ logs=[]; saveLogs(); renderLogs(); }
});
downloadLogsBtn.addEventListener('click', ()=>{
  if(!logs.length){ alert('No logs to download yet.'); return; }
  const headers=['when','label','similarity','confidence','direction'];
  const rows=logs.map(l=>[
    l.when||'',
    (l.label||'').replace(/"/g,'""'),
    (typeof l.similarity==='number' && !Number.isNaN(l.similarity))? l.similarity.toFixed(4):'',
    (typeof l.confidence==='number' && !Number.isNaN(l.confidence))? l.confidence.toFixed(4):'',
    l.direction||''
  ]);
  const csv=[headers.join(','), ...rows.map(r=>r.map(v=>`"${String(v)}"`).join(','))].join('\n');
  const blob=new Blob([csv],{type:'text/csv'}); const url=URL.createObjectURL(blob);
  const a=document.createElement('a'); a.href=url; a.download='sachi-logs.csv'; a.click(); URL.revokeObjectURL(url);
});
resetModelBtn.addEventListener('click', ()=>{
  if(confirm('Reset model and trained sounds?')){ model.labels={}; saveModel(); alert('Reset complete.'); }
});

// debug toggle
if(debugToggle && debugPanel){
  debugToggle.addEventListener('click', ()=>{
    const open = debugPanel.style.display !== 'none';
    debugPanel.style.display = open ? 'none' : 'block';
    debugToggle.textContent = open ? 'Show Raw Top-5' : 'Hide Raw Top-5';
  });
}

// ======================== Start/Stop ========================
listenBtn.addEventListener('click', ()=>{ if(!listening) startListening(); else stopListening(); });

async function startListening(){
  await ensureAudioCtx();
  if(!yamnetModel) await loadYamnet();
  if(CLASS_MAP===null) await loadClassMap();

  try{
    const stream = await navigator.mediaDevices.getUserMedia({audio:true});
    micStream = stream;
    const src = audioContext.createMediaStreamSource(stream);
    setupAnalyser();
    src.connect(analyser);

    // Stereo path for compass
    try{
      splitNode = audioContext.createChannelSplitter(2);
      src.connect(splitNode);
      analyserL = audioContext.createAnalyser();
      analyserR = audioContext.createAnalyser();
      analyserL.fftSize = 1024; analyserR.fftSize = 1024;
      analyserL.smoothingTimeConstant = 0.6; analyserR.smoothingTimeConstant = 0.6;
      splitNode.connect(analyserL, 0);
      splitNode.connect(analyserR, 1);
      timeL = new Float32Array(analyserL.fftSize);
      timeR = new Float32Array(analyserR.fftSize);
    }catch(e){ console.warn('Stereo compass init failed (mono device?):', e); }

    const desiredLen = Math.max(16000, Math.min(48000, Math.round(audioContext.sampleRate*RING_SECONDS)));
    ringBuffer = new Float32Array(desiredLen);
    ringWrite = 0; ringTotalWritten = 0;

    listenBtn.classList.add('active');
    listenStatus.textContent='Listening'; micGround.textContent='Microphone active';
    listening=true; const pulse=document.querySelector('.pulse'); if(pulse) pulse.style.opacity='1';
    if (compassCtx) requestAnimationFrame(animateCompass);
    updateLoop();
  }catch(e){ console.error(e); alert('Microphone access required.'); }
}
function stopListening(){
  listenBtn.classList.remove('active');
  listenStatus.textContent='Idle'; micGround.textContent='Microphone not active';
  micStream && micStream.getTracks().forEach(t=>t.stop()); micStream=null;
  try{ splitNode && splitNode.disconnect(); analyserL && analyserL.disconnect(); analyserR && analyserR.disconnect(); }catch{}
  splitNode = analyserL = analyserR = null;
  rafId && cancelAnimationFrame(rafId); listening=false;
  const pulse=document.querySelector('.pulse'); if(pulse) pulse.style.opacity='0';
}

// ======================== Loop ========================
let lastClassifiedAt = 0;

async function updateLoop(){
  rafId = requestAnimationFrame(updateLoop);
  try{
    drawVisualizer();
    if(!analyser) return;

    // RMS for VAD / label
    analyser.getFloatTimeDomainData(timeDataArray);
    let sum=0; for(let i=0;i<timeDataArray.length;i++){ const x=timeDataArray[i]; sum += x*x; }
    const rms = Math.sqrt(sum / timeDataArray.length);
    levelText.textContent = Math.round(rms*10000) + ' (level)';
    lastEnergyRMS = rms;

    // Stereo energy for compass
    if(analyserL && analyserR && timeL && timeR){
      analyserL.getFloatTimeDomainData(timeL);
      analyserR.getFloatTimeDomainData(timeR);
      let sL=0,sR=0; for(let i=0;i<timeL.length;i++){ sL+=timeL[i]*timeL[i]; sR+=timeR[i]*timeR[i]; }
      const rmsL = Math.sqrt(sL/timeL.length), rmsR = Math.sqrt(sR/timeR.length);
      const eps = 1e-7;
      const ild = (rmsR - rmsL) / (rmsR + rmsL + eps); // -1..+1
      const angle = Math.max(-90, Math.min(90, ild * 90));
      compassAngleTarget = angle;

      if (rms < 0.012) { lastDirLabel='Unknown'; lastAzDeg=0; }
      else if (angle < -15) { lastDirLabel='Left'; lastAzDeg=Math.round(angle); }
      else if (angle > 15) { lastDirLabel='Right'; lastAzDeg=Math.round(angle); }
      else { lastDirLabel='Center'; lastAzDeg=Math.round(angle); }

      if (dirText) dirText.textContent = lastDirLabel;
      if (azText)  azText.textContent  = `${Math.round(angle)}°`;
    }

    const now = performance.now();
    if(now - lastClassifiedAt < CLASSIFY_INTERVAL_MS) return;
    lastClassifiedAt = now;

    // Warm-up + VAD
    if(ringTotalWritten < ringBuffer.length * 0.8) {
      // maintain ring buffer
    } else if(rms < VAD_RMS_GATE) {
      return;
    }

    // maintain ring buffer
    const td = timeDataArray;
    const N=ringBuffer.length, M=td.length;
    const first=Math.min(M, N-ringWrite);
    ringBuffer.set(td.subarray(0,first), ringWrite);
    if(first<M) ringBuffer.set(td.subarray(first), 0);
    ringWrite = (ringWrite + M) % N;
    ringTotalWritten += M;

    // slice ~1s
    const oneSec = new Float32Array(ringBuffer.length);
    const tail = ringBuffer.subarray(ringWrite);
    const head = ringBuffer.subarray(0, ringWrite);
    oneSec.set(tail, 0); oneSec.set(head, tail.length);

    // classify
    const res = await classifyBuffer(oneSec, audioContext.sampleRate);
    if(res.top5) renderTop5(res.top5);

    const threshold = parseFloat(thresholdInput.value || '0.75');
    let pred = { label:null, score:0, friendly:null, confidence:null };

    // prefer strong custom
    if(res.custom?.label && res.custom.score >= threshold){
      pred = { label: res.custom.label, score: res.custom.score, friendly: friendlyFromRawLabel(res.custom.label), confidence: null };
    }

    // fallback to pretrained
    if(!pred.label && res.pretrained?.label){
      const conf = +res.pretrained.score || 0;
      // ---- RELAX: allow Speech even if conf is low when user is clearly speaking (RMS high)
      const isSpeech = /speech|talk|voice/i.test(res.pretrained.label);
      const pass = conf >= PRETRAINED_MIN_CONF || (isSpeech && rms >= SPEECH_RMS_OVERRIDE);
      if(pass){
        pred = { label: res.pretrained.label, score: conf, friendly: res.friendly, confidence: conf };
      }
    }

    // if still nothing, but VAD open: log as Other (so logs aren't empty during activity)
    if(!pred.label){
      pred = { label: 'Other', score: 0, friendly: null, confidence: null };
    }

    const voted = pushAndVote(pred);
    const finalLabel = voted.label;
    const finalScore = voted.score;
    const finalConf = pred.confidence;
    const icon = iconForFriendly(voted.friendly ?? friendlyFromRawLabel(finalLabel) ?? 'Other');

    const direction = `${lastDirLabel}`;
    const entry = { when: nowIso(), label: finalLabel, similarity: finalScore, confidence: finalConf, direction, icon };

    const last = logs[logs.length-1];
    if(!loggingPaused && (!last || last.label!==entry.label || (new Date(entry.when)-new Date(last.when)>3000))){
      logs.push(entry); saveLogs(); renderLogs();
      try{
        listenBtn.animate(
          [{transform:'translateY(0) scale(1)'},{transform:'translateY(-6px) scale(1.03)'},{transform:'translateY(0) scale(1)'}],
          {duration:420, easing:'ease-out'}
        );
      }catch{}
    }

  }catch(e){ console.warn('updateLoop error', e); }
}

// ======================== Startup ========================
thresholdLabel.textContent = thresholdInput.value;
thresholdInput.addEventListener('input', ()=> thresholdLabel.textContent = thresholdInput.value);
modelSelector?.addEventListener('change', async ()=>{
  yamnetModel=null; modelSummary.textContent='Loading pretrained model...'; await loadYamnet();
});
(async function init(){
  loadModel(); loadLogs(); await loadYamnet(); await loadClassMap();
  modelSummary.textContent='Ready';
  drawVisualizer();
  drawCompass(0,0);
})();
