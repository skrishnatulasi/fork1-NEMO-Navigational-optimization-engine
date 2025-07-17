// --- Navigation ---
document.querySelectorAll('#sidebar button').forEach(btn => {
  btn.addEventListener('click', function() {
    document.querySelectorAll('#sidebar button').forEach(b => b.classList.remove('active'));
    this.classList.add('active');
    document.querySelectorAll('.section').forEach(sec => sec.style.display = 'none');
    document.getElementById('section-' + this.dataset.section).style.display = '';
  });
});

// --- Toasts ---
function showToast(msg, type='success') {
  const toast = document.getElementById('toast');
  const body = document.getElementById('toast-body');
  toast.className = `toast align-items-center text-bg-${type} border-0 position-fixed bottom-0 end-0 m-3`;
  body.textContent = msg;
  toast.style.display = '';
  setTimeout(() => { toast.style.display = 'none'; }, 3000);
}
function hideToast() {
  document.getElementById('toast').style.display = 'none';
}

// --- Map Setup ---
let map, userMarker, userLocation = [11.0, 78.5], hotspotMarkers = [], routeLine = null;
function initMap() {
  map = L.map('map').setView(userLocation, 6);
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; OpenStreetMap contributors'
  }).addTo(map);
  userMarker = L.marker(userLocation, { draggable: true }).addTo(map)
    .bindPopup('Your Location').openPopup();
  userMarker.on('moveend', function(e) {
    userLocation = [e.target.getLatLng().lat, e.target.getLatLng().lng];
  });
  map.on('click', function(e) {
    userLocation = [e.latlng.lat, e.latlng.lng];
    userMarker.setLatLng(e.latlng).openPopup();
  });
}
if (document.getElementById('map')) initMap();

// --- Predict Zones ---
let heatLayer = null;

// --- Fallback Data ---
const fallbackPFZ = [
  {lat: 9.3, lon: 79.1, catch_pred: 12},
  {lat: 9.4, lon: 79.2, catch_pred: 8},
  {lat: 9.5, lon: 79.3, catch_pred: 5},
];
const fallbackDashboard = {
  hotspot_count: 3,
  best_predicted_catch: 12,
  user_env: {sst: 28, wind_speed: 4, salinity: 35}
};
const fallbackRoute = {
  geometry: {coordinates: [[79.3127,9.2885],[79.1,9.3],[79.2,9.4],[79.3,9.5],[79.3127,9.2885]]},
  properties: {distance_km: 20, est_time_hr: 1.5, est_fuel_l: 4}
};

// --- Modified loadPFZ with fallback ---
async function loadPFZ() {
  try {
    hotspotMarkers.forEach(m => map.removeLayer(m));
    hotspotMarkers = [];
    if (heatLayer) map.removeLayer(heatLayer);
    const res = await fetch(`/predict_zones?user_lat=${userLocation[0]}&user_lon=${userLocation[1]}&radius_km=50`);
    let data = await res.json();
    if (!data.features || !data.features.length) throw new Error('No features');
    const heatData = [];
    data.features.forEach(f => {
      const lat = f.geometry.coordinates[1];
      const lon = f.geometry.coordinates[0];
      const catch_pred = f.properties.catch_pred;
      heatData.push([lat, lon, Math.max(0, catch_pred)]);
      const marker = L.circleMarker([lat, lon], {
        radius: 6,
        color: catch_pred > 10 ? 'red' : catch_pred > 5 ? 'orange' : 'blue',
        fillOpacity: 0.7
      }).addTo(map)
        .bindPopup(`<b>Predicted Catch:</b> ${catch_pred.toFixed(2)}<br>Lat: ${lat.toFixed(3)}<br>Lon: ${lon.toFixed(3)}`);
      hotspotMarkers.push(marker);
    });
    heatLayer = L.heatLayer(heatData, {radius: 25, blur: 15, maxZoom: 10}).addTo(map);
    showToast(lang === 'ta' ? 'மீன்பிடி இடங்கள் ஏற்றப்பட்டது!' : 'Predictions loaded!');
    await loadDashboard();
  } catch (e) {
    // Fallback
    hotspotMarkers.forEach(m => map.removeLayer(m));
    hotspotMarkers = [];
    if (heatLayer) map.removeLayer(heatLayer);
    const heatData = [];
    fallbackPFZ.forEach(f => {
      heatData.push([f.lat, f.lon, Math.max(0, f.catch_pred)]);
      const marker = L.circleMarker([f.lat, f.lon], {
        radius: 6,
        color: f.catch_pred > 10 ? 'red' : f.catch_pred > 5 ? 'orange' : 'blue',
        fillOpacity: 0.7
      }).addTo(map)
        .bindPopup(`<b>Predicted Catch:</b> ${f.catch_pred.toFixed(2)}<br>Lat: ${f.lat.toFixed(3)}<br>Lon: ${f.lon.toFixed(3)}`);
      hotspotMarkers.push(marker);
    });
    heatLayer = L.heatLayer(heatData, {radius: 25, blur: 15, maxZoom: 10}).addTo(map);
    showToast(lang === 'ta' ? 'Fallback: கணிப்புகள்!' : 'Fallback: Predictions!','warning');
    await loadDashboard(true);
  }
}

// --- Language Support ---
let lang = 'ta'; // 'ta' for Tamil, 'en' for English
const translations = {
  ta: {
    hotspots: 'சூடான இடங்கள்',
    best_catch: 'சிறந்த மீன்பிடி',
    sst: 'கடல் மேற்பரப்பு வெப்பநிலை',
    wind: 'காற்று வேகம்',
    salinity: 'உப்புத்தன்மை',
    route_distance: 'பயண தூரம்',
    est_time: 'மதிப்பிடப்பட்ட நேரம்',
    est_fuel: 'எரிபொருள் தேவை',
    route_summary: 'பயண முடிவு',
    best_spot: 'சிறந்த இடம்',
    catch_prob: 'மீன் சாத்தியம்',
    high: 'மிக அதிகம்',
    medium: 'நடுத்தர',
    low: 'குறைவு',
    play_voice: '🔊 குரல் அறிவுரை',
    optimize_route: 'வழியை மேம்படுத்து',
    predict_zones: 'மீன்பிடி இடங்களை கணிக்க',
    map_title: 'வரைபடம் மற்றும் மீன்பிடி இடங்கள்',
    summary: '🎯 முடிவு:',
    location: 'இடம்',
    fuel: 'எரிபொருள்',
    time: 'நேரம்',
    km: 'கி.மீ',
    hr: 'மணி',
    l: 'லிட்டர்',
    best: 'சிறந்த',
    sea_conditions: 'கடலின் நிலை',
    best_fishing: 'சிறந்த மீன்பிடி இடம்',
    route_fuel: 'பயண வழி',
    voice_guide: 'குரல் அறிவுரை',
    past_trips: 'எனது கடந்த பயணங்கள்',
  },
  en: {
    hotspots: 'Hotspots',
    best_catch: 'Best Predicted Catch',
    sst: 'Current SST',
    wind: 'Wind',
    salinity: 'Salinity',
    route_distance: 'Route Distance',
    est_time: 'Est. Time',
    est_fuel: 'Est. Fuel',
    route_summary: 'Route Summary',
    best_spot: 'Best Spot',
    catch_prob: 'Catch Probability',
    high: 'High',
    medium: 'Medium',
    low: 'Low',
    play_voice: '🔊 Play Voice Guidance',
    optimize_route: 'Optimize Route',
    predict_zones: 'Predict Zones',
    map_title: 'Map & Potential Fishing Zones',
    summary: '🎯 Summary:',
    location: 'Location',
    fuel: 'Fuel',
    time: 'Time',
    km: 'km',
    hr: 'hr',
    l: 'L',
    best: 'Best',
    sea_conditions: 'Sea Conditions',
    best_fishing: 'Best Fishing Spot',
    route_fuel: 'Route & Fuel Estimation',
    voice_guide: 'Voice Guidance',
    past_trips: 'My Past Trips',
  }
};
function t(key) { return translations[lang][key] || key; }

// --- Dashboard Rendering ---
let lastDashboardData = null;
function renderDashboard() {
  if (!lastDashboardData) return;
  const d = lastDashboardData;
  document.getElementById('dashboard-content').innerHTML = `
    <div class="row">
      <div class="col"><b>${t('hotspots')}:</b> ${d.hotspot_count}</div>
      <div class="col"><b>${t('best_catch')}:</b> ${d.best_predicted_catch.toFixed(2)}</div>
      <div class="col"><b>${t('sst')}:</b> ${d.user_env.sst.toFixed(2)}°C</div>
      <div class="col"><b>${t('wind')}:</b> ${d.user_env.wind_speed.toFixed(2)} m/s</div>
      <div class="col"><b>${t('salinity')}:</b> ${d.user_env.salinity.toFixed(2)}</div>
    </div>
  `;
}

// --- City Selector ---
let selectedCity = 'Rameshwaram';
const cities = ['Rameshwaram', 'Chennai', 'Nagapattinam', 'Tuticorin'];
window.addEventListener('DOMContentLoaded', function() {
  // Add city selector above dashboard
  const dash = document.getElementById('dashboard-content');
  if (dash && !document.getElementById('citySelector')) {
    const sel = document.createElement('select');
    sel.id = 'citySelector';
    sel.className = 'form-select mb-2';
    cities.forEach(c => {
      const opt = document.createElement('option');
      opt.value = c;
      opt.textContent = c;
      sel.appendChild(opt);
    });
    sel.value = selectedCity;
    sel.onchange = function() {
      selectedCity = this.value;
      loadDashboard();
    };
    dash.parentNode.insertBefore(sel, dash);
  }
});

// --- Modified loadDashboard to use city param ---
async function loadDashboard(isFallback) {
  try {
    if (isFallback) throw new Error('Force fallback');
    const res = await fetch(`/dashboard?city=${selectedCity}`);
    const data = await res.json();
    if (!data.user_env) throw new Error('No env');
    lastDashboardData = data;
    renderDashboard();
  } catch (e) {
    lastDashboardData = fallbackDashboard;
    renderDashboard();
  }
}

// --- Route Summary Rendering ---
let lastRouteData = null;
function renderRouteSummary(routeData) {
  lastRouteData = routeData;
  if (!routeData || !routeData.properties) return;
  const props = routeData.properties;
  const coords = routeData.geometry && routeData.geometry.coordinates ? routeData.geometry.coordinates : [];
  let bestSpot = coords.length > 1 ? coords[1] : null;
  let summary = `
    <div class="card mt-3">
      <div class="card-header"><b>${t('summary')}</b></div>
      <div class="card-body">
        <div><b>${t('location')}:</b> ${bestSpot ? bestSpot[1].toFixed(3) + ', ' + bestSpot[0].toFixed(3) : '-'}</div>
        <div><b>${t('catch_prob')}:</b> ${props.distance_km > 30 ? t('high') : props.distance_km > 15 ? t('medium') : t('low')}</div>
        <div><b>${t('fuel')}:</b> ${props.est_fuel_l.toFixed(1)} ${t('l')}</div>
        <div><b>${t('time')}:</b> ${props.est_time_hr.toFixed(1)} ${t('hr')}</div>
        <div><b>Expected Yield:</b> ${props.total_expected_yield ? props.total_expected_yield.toFixed(1) + ' kg' : '-'}</div>
        <div><b>Total Risk:</b> ${props.total_risk !== undefined ? props.total_risk.toFixed(1) : '-'}</div>
        <div class="mt-2"><b>Instructions (English):</b><br><span style="font-size:0.95em">${props.instructions_en || ''}</span></div>
        <div class="mt-2"><b>வழிமுறைகள் (தமிழ்):</b><br><span style="font-size:0.95em">${props.instructions_ta || ''}</span></div>
        <button class="btn btn-warning mt-2 me-2" onclick="playRouteVoice('en')">🔊 Play English</button>
        <button class="btn btn-warning mt-2 me-2" onclick="playRouteVoice('ta')">🔊 தமிழ்</button>
        <button class="btn btn-success mt-2 me-2" onclick="downloadTamilAudio()">⬇️ Download Tamil Audio</button>
        <button class="btn btn-primary mt-2" onclick="playRouteDescription()">🔊 Play Full Audio Description</button>
      </div>
    </div>
  `;
  document.getElementById('route-summary').innerHTML = summary;
}

window.playRouteVoice = function(langCode) {
  if (!lastRouteData || !lastRouteData.properties) return;
  let text = langCode === 'ta' ? lastRouteData.properties.instructions_ta : lastRouteData.properties.instructions_en;
  if (!text) return;
  const utter = new SpeechSynthesisUtterance(text);
  if (langCode === 'ta') utter.lang = 'ta-IN';
  else utter.lang = 'en-US';
  window.speechSynthesis.speak(utter);
}

window.downloadTamilAudio = async function() {
  if (!lastRouteData || !lastRouteData.route) return;
  const res = await fetch('/tamil_voice_route', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({route: lastRouteData.route})
  });
  const data = await res.json();
  if (data.audio_url) {
    const a = document.createElement('a');
    a.href = data.audio_url;
    a.download = 'route_tamil.mp3';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }
}

window.playRouteDescription = function() {
  if (!lastRouteData || !lastRouteData.properties) return;
  let text = lang === 'ta' ? lastRouteData.properties.instructions_ta : lastRouteData.properties.instructions_en;
  if (!text) return;
  const utter = new SpeechSynthesisUtterance(text);
  utter.rate = 0.95;
  utter.lang = lang === 'ta' ? 'ta-IN' : 'en-US';
  window.speechSynthesis.speak(utter);
}

document.getElementById('predictBtn').onclick = loadPFZ;

// --- Route Markers ---
let routeMarkers = [];

function clearRouteMarkers() {
  routeMarkers.forEach(m => map.removeLayer(m));
  routeMarkers = [];
}

function renderRouteOnMap(routeData) {
  if (!routeData || !routeData.route) return;
  if (routeLine) map.removeLayer(routeLine);
  clearRouteMarkers();
  // Always start/end at Rameshwaram
  const rameshwaram = [9.2885, 79.3127];
  let coords = routeData.route.map(pt => [pt.lat, pt.lon]);
  if (coords.length < 2 || coords[0][0] !== rameshwaram[0] || coords[0][1] !== rameshwaram[1]) {
    coords = [rameshwaram, ...coords, rameshwaram];
  }
  routeLine = L.polyline(coords, {
    color: 'green',
    weight: 7,
    opacity: 0.95,
    className: 'route-shadow'
  }).addTo(map);
  // Add markers for each stop (except start/end)
  routeData.route.forEach((pt, i) => {
    if (i === 0 || i === routeData.route.length - 1) return;
    const marker = L.marker([pt.lat, pt.lon], {icon: L.icon({iconUrl: 'https://cdn.jsdelivr.net/npm/leaflet@1.7.1/dist/images/marker-icon.png', iconSize: [25,41], iconAnchor: [12,41]})})
      .addTo(map)
      .bindPopup(`<b>Stop ${i}</b><br>Yield: ${pt.expected_yield ? pt.expected_yield.toFixed(1) : '-'}<br>Lat: ${pt.lat.toFixed(3)}<br>Lon: ${pt.lon.toFixed(3)}`);
    routeMarkers.push(marker);
  });
  // Fit map to route
  if (coords.length > 1) {
    map.fitBounds(coords);
  }
}

// Add CSS for route shadow
if (!document.getElementById('route-shadow-style')) {
  const style = document.createElement('style');
  style.id = 'route-shadow-style';
  style.innerHTML = `.route-shadow { filter: drop-shadow(0px 0px 6px #222); }`;
  document.head.appendChild(style);
}

// --- Modified routeBtn onclick with robust heatmap/route ---
document.getElementById('routeBtn').onclick = async function() {
  try {
    if (hotspotMarkers.length === 0) {
      showToast(lang === 'ta' ? 'முதலில் இடங்களை பெறவும்!' : 'Fetch hotspots first!', 'danger');
      return;
    }
    const res = await fetch(`/predict_zones?user_lat=${userLocation[0]}&user_lon=${userLocation[1]}&radius_km=50`);
    const data = await res.json();
    const hotspots = data.features.map(f => ({
      lat: f.geometry.coordinates[1],
      lon: f.geometry.coordinates[0],
      catch_pred: f.properties.catch_pred
    }));
    const topHotspots = hotspots.slice().sort((a, b) => b.catch_pred - a.catch_pred).slice(0, 5);
    const routeRes = await fetch('/suggest_route', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        user_lat: userLocation[0],
        user_lon: userLocation[1],
        hotspots: topHotspots,
        n: 5
      })
    });
    const routeData = await routeRes.json();
    renderRouteOnMap(routeData);
    renderRouteSummary(routeData);
    showToast(lang === 'ta' ? 'வழி மேம்படுத்தப்பட்டது!' : 'Route optimized!');
  } catch (e) {
    // Fallback
    if (routeLine) map.removeLayer(routeLine);
    clearRouteMarkers();
    const coords = fallbackRoute.geometry.coordinates.map(([lon, lat]) => [lat, lon]);
    routeLine = L.polyline(coords, { color: 'green', weight: 5 }).addTo(map);
    renderRouteSummary(fallbackRoute);
    showToast(lang === 'ta' ? 'Fallback: வழி!' : 'Fallback: Route!','warning');
  }
}

// --- Catch Log ---
document.getElementById('catchForm').onsubmit = async function(e) {
  e.preventDefault();
  const fishType = document.getElementById('fishType').value;
  const weight = document.getElementById('weight').value;
  await fetch('/log_catch', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      user: 'Fisherman',
      lat: 9.2885,
      lon: 79.3127,
      catch: weight,
      species: fishType,
      price: 0,
      buyer: '',
      timestamp: new Date().toISOString()
    })
  });
  alert('Catch logged!');
  document.getElementById('catchForm').reset();
};
async function loadCatchLog() {
  const res = await fetch('/catch_log');
  const data = await res.json();
  document.getElementById('catch-log-list').innerHTML = data.logs.map(
    l => `<div>${l.timestamp}: <b>${l.fishType}</b> - ${l.weight}kg @ ₹${l.price} (${l.buyer})</div>`
  ).join('');
}
loadCatchLog();

// --- Income Tracker ---
async function loadIncome() {
  const res = await fetch('/income_tracker');
  const data = await res.json();
  document.getElementById('income-content').innerHTML = `
    <div><b>This Month:</b> ₹${data.month}</div>
    <div><b>This Week:</b> ₹${data.week}</div>
    <div><b>Today:</b> ₹${data.today}</div>
  `;
}
loadIncome();

// --- Harbor Status ---
async function loadHarbor() {
  const res = await fetch('/harbor_status');
  const data = await res.json();
  document.getElementById('harbor-content').innerHTML = `<b>Status:</b> ${data.status}<br><b>Fuel:</b> ${data.fuel}<br><b>Ice:</b> ${data.ice}`;
}
loadHarbor();

// --- Subsidy Checker ---
async function loadSubsidy() {
  const res = await fetch('/subsidy_checker');
  const data = await res.json();
  document.getElementById('subsidy-content').innerHTML = `<b>Eligible:</b> ${data.eligible ? 'Yes' : 'No'}<br><b>Schemes:</b> ${data.schemes}`;
}
loadSubsidy();

// --- Safety & SOS ---
document.getElementById('safety-content').innerHTML = `
  <button class="btn btn-danger" onclick="showToast('SOS sent! (Demo)', 'danger')"><i class="fa-solid fa-triangle-exclamation"></i> Send SOS</button>
  <div class="mt-2">Geofencing and emergency alerts coming soon.</div>
`;

// --- Language Toggle (Demo) ---
document.getElementById('langBtn').onclick = function() {
  lang = lang === 'ta' ? 'en' : 'ta';
  renderDashboard();
  renderRouteSummary(lastRouteData);
  this.innerHTML = lang === 'ta' ? '🌐 தமிழ்' : '🌐 English';
};

async function playTamilAudio(route) {
  const res = await fetch('/tamil_voice_route', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({route})
  });
  const data = await res.json();
  const audio = new Audio(data.audio_url);
  audio.play();
}

// --- Simulate Fisherman Optimization ---
async function simulateFisherman() {
  lang = 'ta'; // Default to Tamil
  userLocation = [9.2885, 79.3127]; // Rameshwaram
  if (map && userMarker) {
    userMarker.setLatLng({lat: userLocation[0], lng: userLocation[1]}).openPopup();
    map.setView(userLocation, 8);
  }
  await loadPFZ();
  await new Promise(r => setTimeout(r, 1000)); // Wait for heatmap
  document.getElementById('routeBtn').click();
}

// Remove Simulate Fisherman button logic
window.addEventListener('DOMContentLoaded', function() {
  // Only auto-run simulation on first load, do not add the button
  simulateFisherman();
});

// --- Live Dashboard for SST, Wind, Storm Alerts (Simulated) ---
let liveAlertInterval = null;
function startLiveDashboard() {
  if (liveAlertInterval) clearInterval(liveAlertInterval);
  const dash = document.getElementById('dashboard-content');
  if (!document.getElementById('live-alert')) {
    const alertDiv = document.createElement('div');
    alertDiv.id = 'live-alert';
    dash.parentNode.insertBefore(alertDiv, dash);
  }
  async function updateLiveAlert() {
    // Simulate live data
    const sst = 27.5 + Math.random() * 2;
    const wind = 3.5 + Math.random() * 3;
    const storm = Math.random() < 0.15; // 15% chance of storm
    let alertHtml = `<b>SST:</b> ${sst.toFixed(1)}°C &nbsp; <b>Wind:</b> ${wind.toFixed(1)} m/s`;
    if (storm) {
      alertHtml += '<div class="alert alert-danger mt-2">⚠️ Storm predicted! Avoid sea travel.</div>';
    } else if (wind > 6) {
      alertHtml += '<div class="alert alert-warning mt-2">⚠️ High wind! Caution advised.</div>';
    }
    document.getElementById('live-alert').innerHTML = alertHtml;
  }
  updateLiveAlert();
  liveAlertInterval = setInterval(updateLiveAlert, 30000);
}
window.addEventListener('DOMContentLoaded', startLiveDashboard);
