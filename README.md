NEMO -NAVIGATIONAL ENGINE USING MARINE OPTIMIZATION


 Transforming Coastal Livelihoods with AI-Powered Precision Fishing
Problem: 
Coastal fishermen in Tamil Nadu face declining fish stocks, unpredictable weather, and fuel costs. Traditional fishing is based on intuition, not data — leading to wasted trips, low yields, and financial insecurity.  

Solution:
Our AI-powered application leverages satellite data (SST, chlorophyll, wind), fleet behavior, and regulatory boundaries to help Tamil Nadu’s fishermen identify profitable,safe, and fuel-efficient,fishing routes — spoken in Tamil, optimized for non-literate users, and visualized on an interactive map.

---

 Why It Matters

Data-Driven Fishing: Replaces guesswork with science-backed predictions.
Fuel Savings:Optimized routes reduce fuel consumption by up to 40%.
Livelihood Upliftment: Increases catch probability, income, and safety.
Zero-Literacy Friendly: Tamil audio guidance makes it fully accessible.
Uses Public Satellite & Fleet Data: Affordable, scalable, and community-first.



 Core Features

| Feature                         | Description                                                                 |
|-------------------------------|-----------------------------------------------------------------------------|
| Fish Yield Prediction  | ML model trained on SST, chlorophyll, fleet data to predict hot fishing zones |
| Route Optimization    | Algorithm computes shortest safe path to high-yield zones under 40km        |
| GFW Fleet Integration  | Visualizes daily fishing fleet activity from `.csv` uploaded data           |
| Tamil Audio Assistant  | Generates spoken directions in Tamil for every route                        |
| Risk Area Filtering   | Filters zones with bad weather, overfished areas, or marine restrictions     |
| Mobile Ready        | Flask-based backend ready for Android/iOS wrappers                         |

---

 Tech Stack

| Layer           | Technology                          |
|----------------|-------------------------------------|
| Backend         | Python + Flask                      |
| AI Model        | Scikit-learn (SST, Chlorophyll, etc.)|
| Data Sources    | NASA, NOAA, GFW Fleet `.csv`        |
| Optimization    | `geopy`, `folium`, `numpy`, `scipy` |
| Voice Engine    | `gTTS` (Tamil)                      |
| Visualization   | `folium`, Leaflet.js                |

---

How It Works – Pipeline

1. Input
   - User location (lat/lon), fuel range  
   - Uploaded `.csv` of recent fleet activity (daily fleet data)

2. **Prediction Engine**  
   - Predicts catch scores on a 10km grid  
   - Combines satellite + GFW historical patterns

3. **Optimizer**  
   - Filters by safe zones, computes distance-cost  
   - Selects zone with best yield-to-fuel ratio

4. **Output**  
   - Route map with Tamil voice instruction  
   - Safe return route with alerts and markers

---

 Example Use Case
 a fisherman from Nagapattinam, starts his app near the coast with a fuel range of 30 km.

- The app suggests a path 22 km east with a **0.91 yield score**
- Tamil voice output:  
  _“மீனவர்கள் சகோதரரே, கிழக்கு திசைக்கு 22 கி.மீ பயணியுங்கள். அதிகமான மீன் வாய்ப்பு உள்ளது.”_
- He returns with 3x more catch than usual — saving fuel and avoiding storms.

- For any queries contact- anjanarangarajan06@gmail.com
- TEAM NEXUS:
- ANJANA RANGARAJAN
- KRISHNA TULASI S



