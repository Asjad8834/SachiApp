// SACHI v1.2.4 — complete JS
// Fixes: reliable frequency beam, VAD gate, temporal smoothing, confidence in logs, Top-5 debug.
// Keeps your sachi code 2 structure and local YAMNet model loading order.

//////////////////////// Selectors ////////////////////////
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

//////////////////////// Constants / State ////////////////////////
const MODEL_KEY = 'sachi_model_v1';
const LOGS_KEY = 'sachi_logs_v1';

const TFHUB_YAMNET_URL = 'https://tfhub.dev/google/tfjs-model/yamnet/classification/1/default/1';
const LOCAL_YAMNET_PATH = './models/yamnet/model.json';
const CLASS_MAP_URL = './models/yamnet/class_map.json';

const MIN_YAMNET_LEN = 15600;     // ~0.975s at 16kHz
const TARGET_SR = 16000;
const RING_SECONDS = 1.0;         // capture window
const CLASSIFY_INTERVAL_MS = 900; // throttle

const PRETRAINED_MIN_CONF = 0.15; // minimum YAMNet confidence
const VAD_RMS_GATE = 0.015;       // silence gate
const SMOOTH_WINDOW = 5;          // majority vote window
const FREQ_BINS_TARGET = 64;

let yamnetModel = null;
let CLASS_MAP = null;             // array of class names by index

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

let model = { labels: {} };       // custom-label cosine-sim vectors

// ring buffer for 1s audio
let ringBuffer = new Float32Array(16000);
let ringWrite = 0;
let ringTotalWritten = 0;

// icon helpers
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
  { pattern: /speech|conversation|talk|narration|whisper/ig, label: 'Speech' },
  { pattern: /car|vehicle|automobile|engine|horn|traffic/ig, label: 'Vehicle' },
  { pattern: /dog|bark|woof/ig, label: 'Dog' },
  { pattern: /bell|doorbell|chime/ig, label: 'Bell' },
  { pattern: /siren|ambulance|police|fire truck/ig, label: 'Siren' },
  { pattern: /music|sing|song|melody|instrument/ig, label: 'Music' }
];

//////////////////////// Utils ////////////////////////
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
  if(l.includes('speech')||l.includes('talk')) return iconMap.Speech;
  if(l.includes('car')||l.includes('vehicle')||l.includes('engine')) return iconMap.Vehicle;
  if(l.includes('dog')||l.includes('bark')) return iconMap.Dog;
  if(l.includes('bell')||l.includes('doorbell')||l.includes('chime')) return iconMap.Bell;
  if(l.includes('siren')) return iconMap.Siren;
  if(l.includes('music')||l.includes('sing')) return iconMap.Music;
  return iconMap.Other;
}

//////////////////////// Persistence ////////////////////////
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

//////////////////////// UI ////////////////////////
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

//////////////////////// Visualizer — frequency beam (fixed) ////////////////////////
const canvas = freqCanvas;
const ctx = canvas.getContext('2d');

// Force real pixel size even if CSS height collapses
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
  // background
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

//////////////////////// Model loading (keeps your order) ////////////////////////
async function loadYamnet(){
  modelSummary.textContent = 'Loading pretrained model...';

  // 1) Local GraphModel
  try{
    yamnetModel = await tf.loadGraphModel(LOCAL_YAMNET_PATH);
    yamnetModel.__isLocal = true;
    console.info('[YAMNet] local GraphModel loaded');
    modelSummary.textContent = 'Local YAMNet TFJS GraphModel loaded.';
    return;
  }catch(e){ console.warn('[YAMNet] local load failed, fallback...', e); }

  // 2) npm model
  try{
    if(typeof yamnet === 'undefined' || !yamnet){
      await loadScriptDynamic('https://cdn.jsdelivr.net/npm/@tensorflow-models/yamnet@1.0.2/dist/yamnet.min.js');
      await new Promise(r=>setTimeout(r,100));
    }
    if(yamnet && typeof yamnet.load==='function'){
      yamnetModel = await yamnet.load();
      console.info('[YAMNet] npm model loaded');
      modelSummary.textContent = 'Pretrained model loaded (YAMNet)';
      return;
    }
  }catch(e){ console.warn('[YAMNet] npm load failed', e); }

  // 3) TFHub GraphModel
  try{
    yamnetModel = await tf.loadGraphModel(TFHUB_YAMNET_URL, { fromTFHub:true });
    yamnetModel.__isTFHubGraphModel = true;
    console.info('[YAMNet] TFHub GraphModel loaded');
    modelSummary.textContent = 'Pretrained model loaded (TFHub)';
  }catch(e){
    console.error('[YAMNet] TFHub load failed', e);
    modelSummary.textContent = 'Failed to load YAMNet — using spectrum fallback';
    yamnetModel = null;
  }
}

async function loadClassMap(){
  try{
    const res = await fetch(CLASS_MAP_URL, { cache:'no-store' });
    if(!res.ok) throw new Error('class_map.json not found');
    const data = await res.json();
    CLASS_MAP = Array.isArray(data) ? data : (Array.isArray(data.classes) ? data.classes : null);
    console.info('[YAMNet] class_map loaded:', CLASS_MAP ? CLASS_MAP.length : 0);
  }catch(e){
    CLASS_MAP = null;
    console.warn('No class_map.json — labels may show as class_###', e);
  }
}

//////////////////////// Audio helpers ////////////////////////
async function ensureAudioCtx(){
  if(!audioContext) audioContext = new (window.AudioContext || window.webkitAudioContext)();
}
function setupAnalyser(){
  if(!analyser){
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 2048;
    analyser.smoothingTimeConstant = 0.8; // stable bars
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
function estimateDirection(){
  if(!micStream || !audioContext) return 'Unknown';
  try{
    const src = audioContext.createMediaStreamSource(micStream);
    const splitter = audioContext.createChannelSplitter(2);
    const aL = audioContext.createAnalyser(), aR = audioContext.createAnalyser();
    aL.fftSize=256; aR.fftSize=256; src.connect(splitter);
    splitter.connect(aL,0); splitter.connect(aR,1);
    const L=new Uint8Array(aL.frequencyBinCount), R=new Uint8Array(aR.frequencyBinCount);
    aL.getByteFrequencyData(L); aR.getByteFrequencyData(R);
    let eL=0,eR=0; for(let i=0;i<L.length;i++){ eL+=L[i]; eR+=R[i]; }
    try{ splitter.disconnect(); aL.disconnect(); aR.disconnect(); }catch{}
    if(eL > eR*1.15) return 'Left'; if(eR > eL*1.15) return 'Right'; return 'Center';
  }catch{ return 'Unknown'; }
}
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

//////////////////////// Classification + Top-5 ////////////////////////
const smoothQueue = []; // {label, score, friendly}
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
          let maxIdx=0; for(let i=1;i<scoresArr.length;i++) if(scoresArr[i]>scoresArr[maxIdx]) maxIdx=i;
          const labelFromMap = (CLASS_MAP && CLASS_MAP[maxIdx]) ? CLASS_MAP[maxIdx] : `class_${maxIdx}`;
          pretrained = { label: labelFromMap, score: scoresArr[maxIdx], rawIndex: maxIdx };

          const idxs = scoresArr.map((v,idx)=>({idx,v})).sort((a,b)=>b.v-a.v).slice(0,5);
          top5 = idxs.map(e=>{
            const nm = (CLASS_MAP && CLASS_MAP[e.idx]) ? CLASS_MAP[e.idx] : `class_${e.idx}`;
            return { label:nm, score:e.v };
          });

          embedding = Array.from(scoresArr);
        }
        t.dispose && t.dispose();
      }else if(typeof yamnetModel.classify==='function'){
        const preds = await yamnetModel.classify(waveform);
        if(Array.isArray(preds) && preds.length){
          const top = preds.reduce((a,b)=> (b.score>a.score?b:a));
          pretrained = { label: top.className || 'Other', score: +top.score || 0 };
          top5 = preds.slice(0,5).map(p=>({label:p.className, score:+p.score||0}));
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

//////////////////////// Capture & Train ////////////////////////
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

//////////////////////// Start/Stop ////////////////////////
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

    const desiredLen = Math.max(16000, Math.min(48000, Math.round(audioContext.sampleRate*RING_SECONDS)));
    ringBuffer = new Float32Array(desiredLen);
    ringWrite = 0; ringTotalWritten = 0;

    listenBtn.classList.add('active');
    listenStatus.textContent='Listening'; micGround.textContent='Microphone active';
    listening=true; const pulse=document.querySelector('.pulse'); if(pulse) pulse.style.opacity='1';
    updateLoop();
  }catch(e){ console.error(e); alert('Microphone access required.'); }
}
function stopListening(){
  listenBtn.classList.remove('active');
  listenStatus.textContent='Idle'; micGround.textContent='Microphone not active';
  micStream && micStream.getTracks().forEach(t=>t.stop()); micStream=null;
  rafId && cancelAnimationFrame(rafId); listening=false;
  const pulse=document.querySelector('.pulse'); if(pulse) pulse.style.opacity='0';
}

//////////////////////// Loop (drawing + VAD + smoothing) ////////////////////////
let lastClassifiedAt = 0;

async function updateLoop(){
  rafId = requestAnimationFrame(updateLoop);
  try{
    // Always draw the frequency beam
    drawVisualizer();

    if(!analyser) return;

    // RMS energy for VAD / level label
    analyser.getFloatTimeDomainData(timeDataArray);
    let sum=0; for(let i=0;i<timeDataArray.length;i++){ const x=timeDataArray[i]; sum += x*x; }
    const rms = Math.sqrt(sum / timeDataArray.length);
    levelText.textContent = Math.round(rms*10000) + ' (level)';

    // maintain ring buffer
    const td = timeDataArray;
    const N=ringBuffer.length, M=td.length;
    const first=Math.min(M, N-ringWrite);
    ringBuffer.set(td.subarray(0,first), ringWrite);
    if(first<M) ringBuffer.set(td.subarray(first), 0);
    ringWrite = (ringWrite + M) % N;
    ringTotalWritten += M;

    const now = performance.now();
    if(now - lastClassifiedAt < CLASSIFY_INTERVAL_MS) return;
    lastClassifiedAt = now;

    // warm-up and VAD
    if(ringTotalWritten < ringBuffer.length * 0.8) return;
    if(rms < VAD_RMS_GATE) return;

    // slice 1s
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
    // else fallback to pretrained if confident
    else if(res.pretrained?.label && (res.pretrained.score || 0) >= PRETRAINED_MIN_CONF){
      pred = { label: res.pretrained.label, score: +res.pretrained.score || 0, friendly: res.friendly, confidence: +res.pretrained.score || 0 };
    } else {
      return; // too weak
    }

    const voted = pushAndVote(pred);
    const finalLabel = voted.label;
    const finalScore = voted.score;
    const finalConf = pred.confidence;
    const icon = iconForFriendly(voted.friendly ?? friendlyFromRawLabel(finalLabel) ?? 'Other');

    if(finalLabel){
      const direction = estimateDirection();
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
    }
  }catch(e){ console.warn('updateLoop error', e); }
}

//////////////////////// Startup ////////////////////////
thresholdLabel.textContent = thresholdInput.value;
thresholdInput.addEventListener('input', ()=> thresholdLabel.textContent = thresholdInput.value);
modelSelector?.addEventListener('change', async ()=>{
  yamnetModel=null; modelSummary.textContent='Loading pretrained model...'; await loadYamnet();
});
(async function init(){
  loadModel(); loadLogs(); await loadYamnet(); await loadClassMap();
  modelSummary.textContent='Ready';
  drawVisualizer(); // paint once before listening
})();
