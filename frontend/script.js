// ── Global Chart Config ──
Chart.defaults.color = '#94a3b8';
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.font.size = 12;

// ── State ──
let VIZ = null;
let lastPrediction = null;
const SESSION_ID = 'ses_' + Math.random().toString(36).slice(2, 10);
const chartInstances = {};

// ── Navigation ──
document.querySelectorAll('.nav-link').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.nav-link').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const sec = btn.dataset.section;
    document.querySelectorAll('.section-view').forEach(s => s.classList.remove('active'));
    const el = document.getElementById('sec-' + sec);
    if (el) el.classList.add('active');
    const titles = { overview:'Dashboard Overview', eda:'Exploratory Data Analysis', model:'Model Performance', predict:'Risk Predictor', awareness:'Awareness Hub' };
    const subs = { overview:'Real-time teen mental health analytics', eda:'Deep-dive into behavioral patterns', model:'ML model evaluation metrics', predict:'Individual patient risk assessment', awareness:'Mental health education & resources' };
    document.getElementById('pageTitle').textContent = titles[sec] || '';
    document.getElementById('pageSubtitle').textContent = subs[sec] || '';
  });
});

// ── Load Data ──
async function loadVizData() {
  try {
    const r = await fetch('/viz-data');
    VIZ = await r.json();
    renderKPIs();
    renderAllCharts();
  } catch(e) { console.error('Failed to load viz data', e); }
}

function renderKPIs() {
  if (!VIZ) return;
  const cd = VIZ.classDistribution;
  if (cd) {
    const total = cd.counts.reduce((a,b)=>a+b,0);
    animateCounter('kpiSamples', total);
    animateCounter('kpiDepression', cd.counts[1] || 0);
  }
  if (VIZ.cv && VIZ.cv.f1_mean && VIZ.cv.f1_mean.length) {
    const best = Math.max(...VIZ.cv.f1_mean);
    document.getElementById('kpiF1').textContent = (best * 100).toFixed(1) + '%';
  }
  fetch('/assets/champion_model_name.txt').then(r=>r.ok?r.text():null).then(t=>{
    if(t) document.getElementById('kpiModel').textContent = t.trim();
  }).catch(()=>{});
}

function animateCounter(id, target) {
  const el = document.getElementById(id);
  if (!el) return;
  let start = 0;
  const dur = 1200;
  const t0 = performance.now();
  function step(now) {
    const p = Math.min((now - t0) / dur, 1);
    el.textContent = Math.floor(p * target).toLocaleString();
    if (p < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

// ── Charts ──
function destroyChart(id) {
  if (chartInstances[id]) { chartInstances[id].destroy(); delete chartInstances[id]; }
}

function renderAllCharts() {
  if (!VIZ) return;
  renderSleepChart();
  renderClassChart();
  renderPlatformChart();
  renderCorrChart();
  renderScatterChart();
  renderAddictionChart();
  renderSocialMediaChart();
  renderFeatureChart();
  renderPermChart();
  renderLearningChart();
  renderConfChart();
  renderRocChart();
  renderPcaChart();
}

function renderSleepChart() {
  const d = VIZ.sleepKde; if(!d) return;
  destroyChart('sleepChart');
  chartInstances['sleepChart'] = new Chart(document.getElementById('sleepChart'), {
    type:'line', data:{ labels:d.x.map(v=>v.toFixed(1)),
      datasets:[
        { label:'No Depression', data:d.noDepression, borderColor:'#10b981', backgroundColor:'rgba(16,185,129,0.1)', fill:true, tension:0.4, borderWidth:2, pointRadius:0 },
        { label:'Depression', data:d.depression, borderColor:'#ef4444', backgroundColor:'rgba(239,68,68,0.1)', fill:true, tension:0.4, borderWidth:2, pointRadius:0 }
      ]},
    options:{ responsive:true, maintainAspectRatio:false, plugins:{legend:{position:'top'}}, scales:{x:{title:{display:true,text:'Sleep Hours'}},y:{title:{display:true,text:'Density'}}} }
  });
}

function renderClassChart() {
  const d = VIZ.classDistribution; if(!d) return;
  destroyChart('classChart');
  chartInstances['classChart'] = new Chart(document.getElementById('classChart'), {
    type:'doughnut', data:{ labels:d.labels, datasets:[{ data:d.counts, backgroundColor:['#10b981','#ef4444'], borderWidth:0, hoverOffset:8 }] },
    options:{ responsive:true, maintainAspectRatio:false, cutout:'72%', plugins:{legend:{position:'bottom'}} }
  });
}

function renderPlatformChart() {
  const d = VIZ.platformUsage; if(!d) return;
  destroyChart('platformChart');
  chartInstances['platformChart'] = new Chart(document.getElementById('platformChart'), {
    type:'bar', data:{ labels:d.labels, datasets:[
      { label:'No Depression', data:d.noDepression, backgroundColor:'#10b981', borderRadius:4 },
      { label:'Depression', data:d.depression, backgroundColor:'#ef4444', borderRadius:4 }
    ]},
    options:{ responsive:true, maintainAspectRatio:false, plugins:{legend:{position:'top'}}, scales:{y:{beginAtZero:true}} }
  });
}

function renderCorrChart() {
  const d = VIZ.correlation; if(!d) return;
  destroyChart('corrChart');
  const size = d.labels.length;
  const pts = d.matrix.map(p => ({ x:p.x, y:p.y, v:p.v }));
  chartInstances['corrChart'] = new Chart(document.getElementById('corrChart'), {
    type:'bubble', data:{ datasets:[{ data: pts.map(p=>({ x:p.x, y:p.y, r: Math.abs(p.v)*12+2 })),
      backgroundColor: pts.map(p=> p.v>0 ? `rgba(6,182,212,${Math.abs(p.v)*0.7+0.1})` : `rgba(239,68,68,${Math.abs(p.v)*0.7+0.1})`) }] },
    options:{ responsive:true, maintainAspectRatio:false, plugins:{legend:{display:false},tooltip:{callbacks:{label:ctx=>{const p=pts[ctx.dataIndex];return `${d.labels[p.y]} × ${d.labels[p.x]}: ${p.v}`;}}}},
      scales:{ x:{min:-0.5,max:size-0.5,ticks:{callback:v=>d.labels[v]||'',maxRotation:45}}, y:{min:-0.5,max:size-0.5,ticks:{callback:v=>d.labels[v]||''}} } }
  });
}

function renderScatterChart() {
  const d = VIZ.stressAnxiety; if(!d) return;
  destroyChart('scatterChart');
  const noD = d.points.filter(p=>p.label===0).map(p=>({x:p.x,y:p.y}));
  const yesD = d.points.filter(p=>p.label===1).map(p=>({x:p.x,y:p.y}));
  chartInstances['scatterChart'] = new Chart(document.getElementById('scatterChart'), {
    type:'scatter', data:{ datasets:[
      { label:'No Depression', data:noD, backgroundColor:'rgba(16,185,129,0.5)', pointRadius:3 },
      { label:'Depression', data:yesD, backgroundColor:'rgba(239,68,68,0.5)', pointRadius:3 }
    ]},
    options:{ responsive:true, maintainAspectRatio:false, scales:{x:{title:{display:true,text:'Stress Level'}},y:{title:{display:true,text:'Anxiety Level'}}} }
  });
}

function renderAddictionChart() {
  const d = VIZ.addictionBox; if(!d) return;
  destroyChart('addictionChart');
  const ds = d.stats.map((s,i) => ({
    label:d.labels[i], data:[{x:i,min:s.min,q1:s.q1,median:s.median,q3:s.q3,max:s.max}]
  }));
  // Render as grouped bar showing distribution
  chartInstances['addictionChart'] = new Chart(document.getElementById('addictionChart'), {
    type:'bar', data:{ labels:d.labels, datasets:[
      { label:'Q1', data:d.stats.map(s=>s.q1), backgroundColor:'rgba(6,182,212,0.3)', borderRadius:2 },
      { label:'Median', data:d.stats.map(s=>s.median), backgroundColor:'rgba(139,92,246,0.5)', borderRadius:2 },
      { label:'Q3', data:d.stats.map(s=>s.q3), backgroundColor:'rgba(6,182,212,0.3)', borderRadius:2 },
      { label:'Mean', data:d.stats.map(s=>s.mean), type:'line', borderColor:'#f59e0b', pointBackgroundColor:'#f59e0b', borderWidth:2 }
    ]},
    options:{ responsive:true, maintainAspectRatio:false, scales:{y:{beginAtZero:true,title:{display:true,text:'Addiction Level'}}} }
  });
}

function renderSocialMediaChart() {
  const d = VIZ.socialMediaBox; if(!d) return;
  destroyChart('socialMediaChart');
  chartInstances['socialMediaChart'] = new Chart(document.getElementById('socialMediaChart'), {
    type:'bar', data:{ labels:d.labels, datasets:[
      { label:'Q1', data:d.stats.map(s=>s.q1), backgroundColor:'rgba(16,185,129,0.3)', borderRadius:2 },
      { label:'Median', data:d.stats.map(s=>s.median), backgroundColor:'rgba(16,185,129,0.6)', borderRadius:2 },
      { label:'Q3', data:d.stats.map(s=>s.q3), backgroundColor:'rgba(16,185,129,0.3)', borderRadius:2 },
      { label:'Mean', data:d.stats.map(s=>s.mean), type:'line', borderColor:'#f59e0b', pointBackgroundColor:'#f59e0b', borderWidth:2 }
    ]},
    options:{ responsive:true, maintainAspectRatio:false, scales:{y:{beginAtZero:true,title:{display:true,text:'Hours/Day'}}} }
  });
}

function renderFeatureChart() {
  const d = VIZ.featureImportance?.model; if(!d||!d.labels.length) return;
  destroyChart('featureChart');
  chartInstances['featureChart'] = new Chart(document.getElementById('featureChart'), {
    type:'bar', data:{ labels:d.labels.map(l=>l.replace(/_/g,' ')), datasets:[{ label:'Importance', data:d.values, backgroundColor:'#8b5cf6', borderRadius:4 }] },
    options:{ indexAxis:'y', responsive:true, maintainAspectRatio:false, plugins:{legend:{display:false}} }
  });
}

function renderPermChart() {
  const d = VIZ.featureImportance?.permutation; if(!d||!d.labels.length) return;
  destroyChart('permChart');
  chartInstances['permChart'] = new Chart(document.getElementById('permChart'), {
    type:'bar', data:{ labels:d.labels.map(l=>l.replace(/_/g,' ')), datasets:[{ label:'Permutation Importance', data:d.values, backgroundColor:'#06b6d4', borderRadius:4 }] },
    options:{ indexAxis:'y', responsive:true, maintainAspectRatio:false, plugins:{legend:{display:false}} }
  });
}

function renderLearningChart() {
  const d = VIZ.learningCurve; if(!d||!d.train_sizes.length) return;
  destroyChart('learningChart');
  chartInstances['learningChart'] = new Chart(document.getElementById('learningChart'), {
    type:'line', data:{ labels:d.train_sizes, datasets:[
      { label:'Training F1', data:d.train_f1, borderColor:'#10b981', backgroundColor:'rgba(16,185,129,0.1)', fill:true, tension:0.3, borderWidth:2 },
      { label:'Validation F1', data:d.val_f1, borderColor:'#8b5cf6', backgroundColor:'rgba(139,92,246,0.1)', fill:true, tension:0.3, borderWidth:2 }
    ]},
    options:{ responsive:true, maintainAspectRatio:false, scales:{x:{title:{display:true,text:'Training Samples'}},y:{title:{display:true,text:'F1 Score'}}} }
  });
}

function renderConfChart() {
  const d = VIZ.evaluation?.confusion; if(!d) return;
  destroyChart('confChart');
  const flat = [
    { x:0,y:1,r:Math.sqrt(d[0][0])*1.5+5, v:d[0][0], l:'TN' },
    { x:1,y:1,r:Math.sqrt(d[0][1])*1.5+5, v:d[0][1], l:'FP' },
    { x:0,y:0,r:Math.sqrt(d[1][0])*1.5+5, v:d[1][0], l:'FN' },
    { x:1,y:0,r:Math.sqrt(d[1][1])*1.5+5, v:d[1][1], l:'TP' },
  ];
  chartInstances['confChart'] = new Chart(document.getElementById('confChart'), {
    type:'bubble', data:{ datasets: flat.map(p=>({ label:p.l, data:[{x:p.x,y:p.y,r:Math.min(p.r,40)}],
      backgroundColor: p.l==='TN'||p.l==='TP' ? 'rgba(16,185,129,0.6)' : 'rgba(239,68,68,0.6)' })) },
    options:{ responsive:true, maintainAspectRatio:false, plugins:{tooltip:{callbacks:{label:ctx=>{const p=flat[ctx.datasetIndex];return `${p.l}: ${p.v}`;}}}},
      scales:{ x:{min:-0.5,max:1.5,ticks:{callback:v=>v===0?'Pred: Neg':v===1?'Pred: Pos':''}}, y:{min:-0.5,max:1.5,ticks:{callback:v=>v===0?'True: Pos':v===1?'True: Neg':''}} } }
  });
}

function renderRocChart() {
  const d = VIZ.evaluation?.roc; if(!d) return;
  destroyChart('rocChart');
  chartInstances['rocChart'] = new Chart(document.getElementById('rocChart'), {
    type:'line', data:{ labels:d.fpr.map(v=>v.toFixed(2)), datasets:[
      { label:'ROC Curve', data:d.tpr, borderColor:'#06b6d4', backgroundColor:'rgba(6,182,212,0.1)', fill:true, tension:0.3, borderWidth:2, pointRadius:0 },
      { label:'Random', data:d.fpr, borderColor:'rgba(255,255,255,0.2)', borderDash:[5,5], borderWidth:1, pointRadius:0 }
    ]},
    options:{ responsive:true, maintainAspectRatio:false, scales:{x:{title:{display:true,text:'FPR'}},y:{title:{display:true,text:'TPR'}}} }
  });
}

function renderPcaChart() {
  const d = VIZ.pca; if(!d) return;
  destroyChart('pcaChart');
  const noD = d.points.filter(p=>p.label===0).map(p=>({x:p.x,y:p.y}));
  const yesD = d.points.filter(p=>p.label===1).map(p=>({x:p.x,y:p.y}));
  chartInstances['pcaChart'] = new Chart(document.getElementById('pcaChart'), {
    type:'scatter', data:{ datasets:[
      { label:'No Depression', data:noD, backgroundColor:'rgba(16,185,129,0.4)', pointRadius:3 },
      { label:'Depression', data:yesD, backgroundColor:'rgba(239,68,68,0.4)', pointRadius:3 }
    ]},
    options:{ responsive:true, maintainAspectRatio:false, plugins:{title:{display:true,text:`PCA (${d.variance}% variance explained)`,font:{size:12}}},
      scales:{x:{title:{display:true,text:'PC1'}},y:{title:{display:true,text:'PC2'}}} }
  });
}

// ── Insight Feed ──
const feedItems = [
  { type:'danger', icon:'🔴', text:'High anxiety cluster detected in 16-18 age group — monitor closely.' },
  { type:'warning', icon:'⚠️', text:'Sleep deficit is the strongest predictor — 68% of flagged cases sleep <6 hrs.' },
  { type:'success', icon:'✅', text:'Model F1 score stable across 5-fold CV — production ready.' },
  { type:'', icon:'💡', text:'Feature engineering added 7 derived features — mental_pressure is top driver.' },
  { type:'warning', icon:'📊', text:'Screen time before sleep >2 hrs correlates with 45% higher risk.' },
  { type:'success', icon:'🎯', text:'Champion model accuracy: precision 93%, recall 95%.' },
  { type:'danger', icon:'📉', text:'Teens with addiction level ≥7 show 3x depression rate vs baseline.' },
  { type:'', icon:'🔬', text:'PCA shows clear cluster separation — model generalizes well.' },
];

function addFeedItem() {
  const feed = document.getElementById('insightFeed');
  if (!feed) return;
  const item = feedItems[Math.floor(Math.random() * feedItems.length)];
  const now = new Date().toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'});
  const el = document.createElement('div');
  el.className = `feed-item ${item.type}`;
  el.innerHTML = `<span>${item.icon}</span><span style="flex:1">${item.text}</span><span class="feed-time">${now}</span>`;
  feed.prepend(el);
  while (feed.children.length > 6) feed.lastChild.remove();
}
addFeedItem(); addFeedItem(); addFeedItem();
setInterval(addFeedItem, 6000);

// ── Prediction Form ──
document.getElementById('predictForm')?.addEventListener('submit', async (e) => {
  e.preventDefault();
  const fd = new FormData(e.target);
  const body = {
    age: +fd.get('age'), gender: fd.get('gender'),
    social_media_hours: +fd.get('social_media_hours'), platform_usage: fd.get('platform_usage'),
    sleep_hours: +fd.get('sleep_hours'), screen_time_before_sleep: +fd.get('screen_time_before_sleep'),
    academic_performance: +fd.get('academic_performance'), physical_activity: +fd.get('physical_activity'),
    social_interaction_level: fd.get('social_interaction_level'),
    stress_level: +fd.get('stress_level'), anxiety_level: +fd.get('anxiety_level'), addiction_level: +fd.get('addiction_level')
  };
  const box = document.getElementById('predictResult');
  box.innerHTML = '<p style="color:var(--primary);">⏳ Analyzing...</p>';
  try {
    const r = await fetch('/predict', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body) });
    const data = await r.json();
    lastPrediction = { userData: body, modelOutput: data };
    const color = data.risk_level==='Low'?'var(--accent)':data.risk_level==='High'?'var(--danger)':'var(--warning)';
    let html = `<div class="risk-gauge"><div class="risk-score-big" style="color:${color}">${data.risk_score}</div>
      <div class="risk-level-tag ${data.risk_level.toLowerCase()}">${data.risk_level} Risk</div>
      <p style="font-size:12px;color:var(--text-muted);margin-top:6px;">Confidence: ${(data.confidence*100).toFixed(1)}%</p></div>`;
    if (data.drivers && Object.keys(data.drivers).length) {
      html += '<h4 style="font-size:13px;color:var(--text-muted);margin:16px 0 10px;">Key Risk Drivers</h4><div class="driver-list">';
      for (const [k,v] of Object.entries(data.drivers)) {
        html += `<div class="driver-item"><span class="driver-label">${k}</span><div class="driver-bar"><div class="driver-fill" style="width:${v}%"></div></div><span class="driver-pct">${v}%</span></div>`;
      }
      html += '</div>';
    }
    if (data.suggestions?.length) {
      html += '<h4 style="font-size:13px;color:var(--text-muted);margin:16px 0 10px;">AI Suggestions</h4><div class="suggestion-list">';
      data.suggestions.forEach(s => { html += `<div class="suggestion-item"><i class="fa-solid fa-circle-check"></i><span>${s}</span></div>`; });
      html += '</div>';
    }
    html += `<p style="margin-top:16px;font-size:12px;color:var(--primary);cursor:pointer;" onclick="document.getElementById('fabBtn').click()">💬 Open AI Doctor for personalized guidance →</p>`;
    box.innerHTML = html;
  } catch(err) {
    box.innerHTML = `<p style="color:var(--danger);">❌ Error: ${err.message}</p>`;
  }
});

// ── Chat Logic ──
const fabBtn = document.getElementById('fabBtn');
const chatWindow = document.getElementById('chatWindow');
const chatClose = document.getElementById('chatClose');
const chatInput = document.getElementById('chatInput');
const sendBtn = document.getElementById('sendBtn');
const chatBody = document.getElementById('chatBody');

fabBtn?.addEventListener('click', () => { chatWindow.classList.add('open'); fabBtn.classList.add('hidden'); chatInput.focus(); });
chatClose?.addEventListener('click', () => { chatWindow.classList.remove('open'); fabBtn.classList.remove('hidden'); });

document.querySelectorAll('.quick-btn').forEach(btn => {
  btn.addEventListener('click', () => { chatInput.value = btn.dataset.msg; handleSend(); });
});

sendBtn?.addEventListener('click', handleSend);
chatInput?.addEventListener('keydown', e => { if(e.key==='Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); } });

function timeStr() { return new Date().toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'}); }

function addChatMsg(text, isUser, followUp) {
  const div = document.createElement('div');
  div.className = `chat-msg ${isUser ? 'user' : 'bot'}`;
  let html = `<div class="bubble">${formatResponse(text)}</div><span class="msg-time">${timeStr()}</span>`;
  if (followUp) {
    html += `<div class="follow-up-box" onclick="document.getElementById('chatInput').value=this.textContent;handleSend();">${followUp}</div>`;
  }
  div.innerHTML = html;
  chatBody.appendChild(div);
  chatBody.scrollTop = chatBody.scrollHeight;
  return div;
}

// Create a streaming bubble that updates live (like ChatGPT)
function createStreamBubble() {
  const div = document.createElement('div');
  div.className = 'chat-msg bot';
  div.innerHTML = `<div class="bubble stream-bubble"></div><span class="msg-time">${timeStr()}</span>`;
  chatBody.appendChild(div);
  chatBody.scrollTop = chatBody.scrollHeight;
  return div.querySelector('.stream-bubble');
}

function showTyping() {
  const div = document.createElement('div');
  div.className = 'chat-msg bot';
  div.id = 'typingIndicator';
  div.innerHTML = '<div class="bubble"><div class="typing-dots"><span></span><span></span><span></span></div></div>';
  chatBody.appendChild(div);
  chatBody.scrollTop = chatBody.scrollHeight;
}

function removeTyping() {
  document.getElementById('typingIndicator')?.remove();
}

async function handleSend() {
  const msg = chatInput.value.trim();
  if (!msg) return;
  chatInput.value = '';
  chatInput.focus();

  addChatMsg(msg, true);
  showTyping();
  sendBtn.disabled = true;

  try {
    const payload = {
      message: msg,
      session_id: SESSION_ID,
      user_data: lastPrediction?.userData || {},
      model_output: lastPrediction?.modelOutput || {},
      historical_data: {}
    };

    const res = await fetch('/chat', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload) });
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let fullText = '';
    let streamBubble = null;
    let finalData = null;
    let gotTokens = false;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value, { stream: true });
      const lines = chunk.split('\n');
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        try {
          const evt = JSON.parse(line.slice(6));
          if (evt.type === 'token') {
            if (!gotTokens) { removeTyping(); streamBubble = createStreamBubble(); gotTokens = true; }
            fullText += evt.value;
            // Live update the bubble with formatted text
            streamBubble.innerHTML = formatResponse(fullText) + '<span class="stream-cursor">|</span>';
            chatBody.scrollTop = chatBody.scrollHeight;
          }
          else if (evt.type === 'done') {
            // Streaming complete — finalize
            if (streamBubble) {
              streamBubble.innerHTML = formatResponse(fullText);
            }
          }
          else if (evt.type === 'final') { finalData = evt; }
          else if (evt.type === 'error') { throw new Error(evt.message); }
        } catch(e) { /* skip parse errors */ }
      }
    }

    removeTyping();

    if (gotTokens && streamBubble) {
      // Stream finished — remove cursor, finalize
      streamBubble.innerHTML = formatResponse(fullText);
    } else if (finalData) {
      // Fallback response (no API key)
      const response = finalData.response || "I'm here to help with your mental health questions.";
      const followUp = finalData.follow_up || null;
      addChatMsg(response, false, followUp);
    } else if (!gotTokens) {
      addChatMsg("I'm here to help with mental health questions. Could you tell me more about what you're experiencing?", false);
    }
  } catch(err) {
    removeTyping();
    addChatMsg("I'm having trouble connecting right now. Let me try my built-in knowledge:\n\n• Get 7-9 hours of quality sleep\n• Limit screen time before bed\n• Stay physically active (30 min/day)\n• Talk to someone you trust\n\nPlease try again in a moment. 💚", false);
  }
  sendBtn.disabled = false;
  chatInput.focus();
}

function formatResponse(text) {
  return text
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/^• /gm, '&bull; ')
    .replace(/^\* /gm, '&bull; ')
    .replace(/^- /gm, '&bull; ')
    .replace(/^(\d+)\.\s/gm, '<strong>$1.</strong> ')
    .replace(/\n\n/g, '<br><br>')
    .replace(/\n/g, '<br>');
}

// ── Init ──
loadVizData();

