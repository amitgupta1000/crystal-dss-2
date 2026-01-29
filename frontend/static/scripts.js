// Prototype scripts: load sample fixtures and populate UI widgets
async function fetchJson(path){try{const r=await fetch(path);if(!r.ok)throw new Error('Fetch failed');return await r.json()}catch(e){console.warn('Could not load',path,e);return null}}

function formatNumber(n){if(n===null||n===undefined) return '—';if(typeof n==='number') return n.toFixed(2);return n}

function getQueryParam(name){const params=new URLSearchParams(window.location.search);return params.get(name)}

async function loadDashboard(){const timeseries=await fetchJson('data/timeseries.json');const models=await fetchJson('data/models.json');const horizon=await fetchJson('data/forecast_horizon.json');const regression=await fetchJson('data/regression_results.json');
  // KPIs
  if(horizon && horizon.length){document.getElementById('nextForecast').textContent=formatNumber(horizon[0].forecast);document.getElementById('kpiLastUpdate').textContent=horizon[0].timestamp}
  if(regression && regression.metrics){document.getElementById('kpiMAE').textContent=formatNumber(regression.metrics.MAE);document.getElementById('kpiMAPE').textContent=formatNumber(regression.metrics.MAPE)}

  // Models list
  const ml=document.getElementById('modelsList');if(models && models.length){ml.innerHTML='';models.forEach(m=>{const li=document.createElement('li');li.innerHTML=`<a href="model_detail.html?model=${encodeURIComponent(m.name)}">${m.name}</a> — v${m.version} — ${m.status}`;ml.appendChild(li)})}

  // Horizon table
  const tbody=document.querySelector('#horizonTable tbody');if(horizon && horizon.length){tbody.innerHTML='';horizon.forEach(r=>{const tr=document.createElement('tr');tr.innerHTML=`<td>${r.timestamp}</td><td>${formatNumber(r.forecast)}</td><td>${formatNumber(r.lower)}</td><td>${formatNumber(r.upper)}</td>`;tbody.appendChild(tr)})}

  // Main chart
  if(timeseries && document.getElementById('mainChart')){
    const ctx=document.getElementById('mainChart').getContext('2d');
    new Chart(ctx,{type:'line',data:{labels:timeseries.map(d=>d.timestamp),datasets:[{label:'Actual',data:timeseries.map(d=>d.actual),borderColor:'#111',fill:false},{label:'Forecast',data:timeseries.map(d=>d.forecast),borderColor:'#2b7cff',fill:false}]},options:{interaction:{mode:'index',intersect:false}}})
  }
  // Sparkline
  if(timeseries && document.getElementById('sparkline')){
    const ctx=document.getElementById('sparkline').getContext('2d');new Chart(ctx,{type:'line',data:{labels:timeseries.slice(-20).map(d=>d.timestamp),datasets:[{data:timeseries.slice(-20).map(d=>d.forecast),borderColor:'#2b7cff',fill:false,pointRadius:0}]},options:{plugins:{legend:{display:false}},elements:{line:{tension:0.3}},scales:{x:{display:false},y:{display:false}}})
  }
}

async function loadModelsPage(){const models=await fetchJson('data/models.json');const list=document.getElementById('modelsListFull');if(models && models.length){list.innerHTML='';models.forEach(m=>{const li=document.createElement('li');li.innerHTML=`<a href="model_detail.html?model=${encodeURIComponent(m.name)}">${m.name}</a> — v${m.version} — ${m.status} — last run: ${m.last_run}`;list.appendChild(li)})}}

async function loadForecastPage(){const ts=await fetchJson('data/timeseries.json');if(!ts || !document.getElementById('forecastChart')) return;const ctx=document.getElementById('forecastChart').getContext('2d');new Chart(ctx,{type:'line',data:{labels:ts.map(d=>d.timestamp),datasets:[{label:'Actual',data:ts.map(d=>d.actual),borderColor:'#111',fill:false},{label:'Forecast',data:ts.map(d=>d.forecast),borderColor:'#2b7cff',fill:false}]},options:{scales:{x:{display:true}}}})}

async function loadModelDetail(){const modelName=getQueryParam('model')||'Acetic Acid';document.getElementById('title').textContent=`Model: ${modelName}`;
  const granger=await fetchJson('data/granger_results.json');const corr=await fetchJson('data/filtered_correlation.json');const regress=await fetchJson('data/regression_results.json');const season=await fetchJson('data/seasonality_summary.json');
  // Quick insight
  const insightEl=document.getElementById('insightText');
  let insights=[];
  if(corr){const top=corr.filter(r=>r.target===modelName).slice(0,3).map(r=>`${r.source} (r=${r.corr.toFixed(2)})`);if(top.length)insights.push(`Top correlates: ${top.join(', ')}`)}
  if(granger){const g=granger.filter(r=>r.Target===modelName).slice(0,3).map(r=>`${r.Source} (p=${r['P-value (F-test)'].toFixed(3)}, lag=${r.Lag})`);if(g.length)insights.push(`Granger sources: ${g.join('; ')}`)}
  if(season){const s=season.find(s=>s.commodity===modelName);if(s)insights.push(`Seasonality: ${s.dominant_period} (strength ${s.strength})`)}
  insightEl.textContent=insights.length?insights.join(' · '):'No automated insights available.';

  // Correlations
  const corrEl=document.getElementById('correlations');if(corr){const rows=corr.filter(r=>r.target===modelName).slice(0,6).map(r=>`<li>${r.source}: ${r.corr.toFixed(2)}</li>`).join('');corrEl.innerHTML=`<h3>Top Correlates</h3><ul>${rows}</ul>`}

  // Causality
  const causEl=document.getElementById('causality');if(granger){const rows=granger.filter(r=>r.Target===modelName && r['P-value (F-test)']<0.05).slice(0,6).map(r=>`<li>${r.Source} → ${r.Target} (p=${r['P-value (F-test)'].toFixed(3)}, lag=${r.Lag})</li>`).join('');causEl.innerHTML=`<h3>Significant Granger Causality</h3><ul>${rows||'<li>None</li>'}</ul>`}

  // Regression
  const regEl=document.getElementById('regression');if(regress && regress.results){const r=regress.results.find(x=>x['Target Commodity']===modelName);if(r){regEl.innerHTML=`<h3>Regression</h3><p>Train Adj R2: ${formatNumber(r['Train Adj R2'])} — Test Adj R2: ${formatNumber(r['Test Adj R2'])}</p><p>Top features: ${r['Features Used'].slice(0,5).join(', ')}</p>`}else regEl.innerHTML='<p>No regression result for this model</p>'}

  // Seasonality
  const seaEl=document.getElementById('seasonality');if(season){const s=season.find(x=>x.commodity===modelName);if(s)seaEl.innerHTML=`<h3>Seasonality</h3><p>Dominant: ${s.dominant_period} — Strength: ${s.strength}</p>`;else seaEl.innerHTML='<p>No seasonality info</p>'}
}

// Init handlers based on page
document.addEventListener('DOMContentLoaded',()=>{if(document.body.classList.contains('ready')) return;const path=location.pathname.split('/').pop();if(path===''||path==='index.html')loadDashboard();else if(path==='models.html')loadModelsPage();else if(path==='forecast.html')loadForecastPage();else if(path==='model_detail.html')loadModelDetail();document.body.classList.add('ready')})
