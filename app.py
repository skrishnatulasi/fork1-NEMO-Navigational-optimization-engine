from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
import joblib
import numpy as np
from geopy.distance import geodesic
from flask_cors import CORS
from scipy.spatial.distance import cdist
import os
import requests
from sklearn.cluster import DBSCAN
from geopy.distance import distance as geodistance
from gtts import gTTS
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

model = joblib.load('models/predictor.pkl')

# Define Rameshwaram and Gulf of Mannar bounding box
SEA_LAT_MIN, SEA_LAT_MAX = 8.9, 9.6
SEA_LON_MIN, SEA_LON_MAX = 78.8, 79.7

def generate_sea_grid(step=0.05):
    lats = np.arange(SEA_LAT_MIN, SEA_LAT_MAX, step)
    lons = np.arange(SEA_LON_MIN, SEA_LON_MAX, step)
    grid = [(lat, lon) for lat in lats for lon in lons]
    return grid

def get_env_features(lat, lon):
    url = (
        f"https://marine-api.open-meteo.com/v1/marine?"
        f"latitude={lat}&longitude={lon}&hourly=sea_surface_temperature,wind_speed_10m"
    )
    try:
        resp = requests.get(url, timeout=3)
        data = resp.json()
        # Check for 'hourly' and required keys
        if (
            'hourly' in data and
            'sea_surface_temperature' in data['hourly'] and
            'wind_speed_10m' in data['hourly'] and
            len(data['hourly']['sea_surface_temperature']) > 0 and
            len(data['hourly']['wind_speed_10m']) > 0
        ):
            sst = data['hourly']['sea_surface_temperature'][0]
            wind = data['hourly']['wind_speed_10m'][0]
            return {
                'sst': sst,
                'salinity': 35,  # fallback or use another API
                'wind_speed': wind,
            }
        else:
            print("Weather API error: missing data in response", data)
            raise ValueError("Missing data in weather API response")
    except Exception as e:
        print("Weather API error:", e)
        # Fallback to simulated data
        return {
            'sst': 25 + np.sin(lat) + np.cos(lon),
            'salinity': 35 + 2 * np.sin(lon),
            'wind_speed': 5 + 3 * np.cos(lat),
        }

@app.route('/')
def serve_frontend():
    return send_from_directory('frontend', 'index.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('frontend/static', filename)

@app.route('/api/tamil_audio', methods=['POST'])
def generate_tamil_audio():
    """Generate Tamil audio for fishing instructions"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"tamil_{timestamp}.mp3"
        filepath = os.path.join('frontend/static/audio', filename)
        
        # Ensure audio directory exists
        os.makedirs('frontend/static/audio', exist_ok=True)
        
        # Generate Tamil TTS
        tts = gTTS(text=text, lang='ta', slow=False)
        tts.save(filepath)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'url': f'/static/audio/{filename}'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_zones', methods=['GET'])
def predict_zones():
    user_lat = float(request.args.get('user_lat', 9.2885))  # Rameshwaram default
    user_lon = float(request.args.get('user_lon', 79.3127))
    radius_km = float(request.args.get('radius_km', 50))
    grid = generate_sea_grid(step=0.05)
    features, coords = [], []
    for lat, lon in grid:
        if geodesic((user_lat, user_lon), (lat, lon)).km <= radius_km:
            env = get_env_features(lat, lon)
            features.append([lat, lon, env['sst'], env['salinity'], env['wind_speed']])
            coords.append((lat, lon))
    if not features:
        return jsonify({"type": "FeatureCollection", "features": []})
    X = pd.DataFrame(features, columns=['latitude', 'longitude', 'sst', 'salinity', 'wind_speed'])
    preds = model.predict(X)
    features_geo = []
    for (lat, lon), pred in zip(coords, preds):
        features_geo.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {"catch_pred": float(pred)}
        })
    # Add Tamil instructions to response
    tamil_instructions = f"இந்த பகுதியில் {len(features)} மீன்பிடி இடங்கள் கண்டுபிடிக்கப்பட்டுள்ளன"
    
    return jsonify({
        "type": "FeatureCollection",
        "features": features_geo,
        "tamil_message": tamil_instructions
    })

def tsp_route(user_loc, hotspots):
    points = [user_loc] + hotspots
    coords = np.array(points)
    dist_matrix = cdist(coords, coords, lambda u, v: geodesic(u, v).km)
    n = len(points)
    visited = [0]
    route = [user_loc]
    while len(visited) < n:
        last = visited[-1]
        next_idx = np.argmin([dist_matrix[last, j] if j not in visited else np.inf for j in range(n)])
        visited.append(next_idx)
        route.append(tuple(coords[next_idx]))
    return route

def route_distance(route_points):
    # route_points: list of (lat, lon)
    total = 0
    for i in range(len(route_points)-1):
        total += geodesic(route_points[i], route_points[i+1]).km
    return total

@app.route('/suggest_route', methods=['POST'])
def route():
    data = request.json
    user_lat = data.get('user_lat')
    user_lon = data.get('user_lon')
    hotspots = data.get('hotspots')
    n = data.get('n', 3)
    optimize_for = data.get('optimize_for', 'fuel')
    if user_lat is None or user_lon is None or not hotspots:
        return jsonify({'error': 'Missing parameters'}), 400
    coords = [(h['lat'], h['lon']) for h in hotspots]
    preds = [h['catch_pred'] for h in hotspots]
    zones = cluster_hotspots(coords, preds)
    if not zones:
        return jsonify({'error': 'No valid fishing zones found'}), 400
    # Optimization logic
    if optimize_for == 'deep-sea':
        # Prefer zones farther from shore
        zones = sorted(zones, key=lambda z: geodesic((user_lat, user_lon), z), reverse=True)
    else:
        # Default: closest/highest catch
        zones = zones
    top_zones = zones[:n]
    route_points = tsp_route((user_lat, user_lon), top_zones)
    total_dist = route_distance(route_points)
    est_time = total_dist / 15
    est_fuel = total_dist * 0.2
    return jsonify({
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": [[lon, lat] for lat, lon in route_points]},
        "properties": {
            "distance_km": total_dist,
            "est_time_hr": est_time,
            "est_fuel_l": est_fuel
        }
    })

@app.route('/log_catch', methods=['POST'])
def log_catch():
    data = request.json
    log_file = 'catch_logs.csv'
    # Ensure all fields exist
    user = data.get('user', 'unknown')
    lat = data.get('lat', 0)
    lon = data.get('lon', 0)
    catch = data.get('catch', data.get('weight', 0))
    species = data.get('species', data.get('fishType', 'unknown'))
    price = data.get('price', 0)
    buyer = data.get('buyer', '')
    timestamp = data.get('timestamp', '')
    with open(log_file, 'a') as f:
        f.write(f"{user},{lat},{lon},{catch},{species},{price},{buyer},{timestamp}\n")
    return jsonify({'status': 'success'})

@app.route('/catch_log', methods=['GET'])
def catch_log():
    log_file = 'catch_logs.csv'
    logs = []
    if os.path.exists(log_file):
        df = pd.read_csv(log_file, header=None, names=['user','lat','lon','catch','species','price','buyer','timestamp'])
        logs = df.to_dict(orient='records')
    return jsonify({'logs': logs})

@app.route('/income_tracker', methods=['GET'])
def income_tracker():
    # Dummy data for now
    return jsonify({'month': 50000, 'week': 12000, 'today': 2000})

@app.route('/harbor_status', methods=['GET'])
def harbor_status():
    # Dummy data for now
    return jsonify({'status': 'Open', 'fuel': 'Available', 'ice': 'Available'})

@app.route('/subsidy_checker', methods=['POST'])
def subsidy_checker():
    data = request.json
    aadhaar = data.get('aadhaar')
    boat = data.get('boat')
    fishing_type = data.get('fishing_type', 'motorized')
    # Dummy logic for demo
    eligible = True if aadhaar and boat else False
    schemes = []
    optimization_hint = ""
    if eligible:
        schemes = ['Fuel Subsidy', 'Insurance']
        if fishing_type == 'motorized':
            optimization_hint = "Optimize for fuel efficiency."
        elif fishing_type == 'deep-sea':
            schemes.append('Gear Subsidy')
            optimization_hint = "Suggest deeper/offshore zones."
    else:
        schemes = []
        optimization_hint = "No subsidy. Warn about higher costs."
    return jsonify({
        'eligible': eligible,
        'schemes': schemes,
        'optimization_hint': optimization_hint
    })

@app.route('/dashboard', methods=['GET'])
def dashboard():
    # Simulate dashboard stats
    user_lat = float(request.args.get('user_lat', 11.0))
    user_lon = float(request.args.get('user_lon', 78.5))
    radius_km = float(request.args.get('radius_km', 50))
    grid = generate_sea_grid(step=0.1)
    features = []
    for lat, lon in grid:
        if geodesic((user_lat, user_lon), (lat, lon)).km <= radius_km:
            env = get_env_features(lat, lon)
            features.append({'lat': lat, 'lon': lon, **env})
    if not features:
        return jsonify({})
    best_catch = max([25 + np.sin(f['lat']) + np.cos(f['lon']) for f in features])
    return jsonify({
        "hotspot_count": len(features),
        "best_predicted_catch": best_catch,
        "user_env": get_env_features(user_lat, user_lon)
    })

def cluster_hotspots(coords, preds, eps_km=10):
    # coords: list of (lat, lon)
    # preds: list of catch_pred
    # Cluster only high-probability points
    high_points = [(lat, lon) for (lat, lon), pred in zip(coords, preds) if pred > 5]
    if not high_points:
        return []
    # Convert lat/lon to meters for clustering
    from sklearn.preprocessing import StandardScaler
    X = np.array(high_points)
    X_scaled = StandardScaler().fit_transform(X)
    clustering = DBSCAN(eps=0.5, min_samples=2).fit(X_scaled)
    labels = clustering.labels_
    zones = []
    for label in set(labels):
        if label == -1:
            continue  # noise
        members = X[labels == label]
        centroid = members.mean(axis=0)
        zones.append(tuple(centroid))
    return zones

def create_grid(center, radius_km, step_km=5):
    lat0, lon0 = center
    points = []
    for dlat in np.arange(-radius_km, radius_km+step_km, step_km):
        for dlon in np.arange(-radius_km, radius_km+step_km, step_km):
            lat = lat0 + dlat / 111
            lon = lon0 + dlon / (111 * np.cos(np.radians(lat0)))
            if geodistance(center, (lat, lon)).km <= radius_km:
                points.append((lat, lon))
    return points

def is_in_banned_zone(point):
    lat, lon = point
    # Example: ban a small area (replace with real logic or shapefile)
    return (9.2 < lat < 9.3) and (79.1 < lon < 79.2)

def is_weather_safe(point):
    # For demo, always True. You can add real logic using get_env_features.
    return True

def score_fn(start, point, yield_score, max_range=60):
    distance_penalty = geodistance(start, point).km / max_range
    return yield_score - 0.5 * distance_penalty

@app.route('/optimized_route', methods=['POST'])
def optimized_route():
    data = request.json
    start_lat = data.get('start_lat', 9.2885)
    start_lon = data.get('start_lon', 79.3127)
    max_range = data.get('max_range', 60)
    n_points = data.get('n_points', 3)
    start = (start_lat, start_lon)
    grid = create_grid(start, radius_km=max_range)
    # Predict yield for each grid point
    features = []
    for lat, lon in grid:
        env = get_env_features(lat, lon)
        features.append([lat, lon, env['sst'], env['salinity'], env['wind_speed']])
    X = pd.DataFrame(features, columns=['latitude', 'longitude', 'sst', 'salinity', 'wind_speed'])
    preds = model.predict(X)
    yield_scores = list(zip(grid, preds))
    # Prune banned/risky locations
    safe_points = [(pt, score) for pt, score in yield_scores if not is_in_banned_zone(pt) and is_weather_safe(pt)]
    # Score + distance optimization
    ranked_points = sorted(
        safe_points,
        key=lambda x: score_fn(start, x[0], x[1], max_range),
        reverse=True
    )
    # Plan route: top n points, round-trip
    route = [start] + [pt for pt, _ in ranked_points[:n_points]] + [start]
    total_distance = sum(geodistance(route[i], route[i+1]).km for i in range(len(route)-1))
    est_fuel = total_distance * 0.2  # Example: 0.2 liters/km
    route_json = [
        {"lat": lat, "lon": lon, "expected_yield": float(score) if i > 0 and i < len(route)-1 else None}
        for i, ((lat, lon), score) in enumerate([(pt, ranked_points[i-1][1]) if i > 0 and i < len(route)-1 else (pt, 0) for i, pt in enumerate(route)])
    ]
    return jsonify({
        "route": route_json,
        "total_distance_km": total_distance,
        "estimated_fuel_use": est_fuel,
        "safe": True
    })

if __name__ == '__main__':
    app.run(debug=True)
