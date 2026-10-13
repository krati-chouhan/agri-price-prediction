import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import joblib
from agri_chatbot import AgriChatbot

app = Flask(__name__)
app.secret_key = 'agriprice-pan-india-ai-secret-key'

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "random_forest_model.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "model_metrics.json")
DATASET_PATH = os.path.join(BASE_DIR, "crops_dataset.csv")

# Load model safely
random_forest_model = None
if os.path.exists(MODEL_PATH):
    try:
        random_forest_model = joblib.load(MODEL_PATH)
        print(f"[OK] Successfully loaded model from {MODEL_PATH}")
    except Exception as e:
        print(f"[WARN] Failed to load model: {e}")
else:
    sub_path = os.path.join(BASE_DIR, "Crop-Price-Prediction-Minor-Project-main", "random_forest_model.joblib")
    if os.path.exists(sub_path):
        random_forest_model = joblib.load(sub_path)
        print(f"[OK] Successfully loaded model from subfolder {sub_path}")

# Load model evaluation metrics
model_metrics = {}
if os.path.exists(METRICS_PATH):
    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            model_metrics = json.load(f)
    except Exception as e:
        print(f"[WARN] Could not load model_metrics.json: {e}")

if not model_metrics:
    model_metrics = {
        "best_model": "Random Forest Regressor",
        "random_forest": {"r2": 0.9874, "rmse": 4.2253, "mae": 2.9595, "mse": 17.8528},
        "adaboost": {"r2": 0.9224, "rmse": 10.481, "mae": 8.9021, "mse": 109.8509},
        "dataset_size": 9600
    }

# Mappings matching trained model
CROP_MAPPING = {
    'Chilli': 0,
    'Groundnut': 1,
    'Maize': 2,
    'Onion': 3,
    'Potato': 4,
    'Pulses': 5,
    'Rice': 6,
    'Sugarcane': 7
}

LOCATION_MAPPING = {
    'Andhra Pradesh': 0,
    'Bihar': 1,
    'Gujarat': 2,
    'Karnataka': 3,
    'Madhya Pradesh': 4,
    'Maharashtra': 5,
    'Punjab': 6,
    'Rajasthan': 7,
    'Uttar Pradesh': 8,
    'West Bengal': 9
}

# Display order emphasizing requested crops (Potato, Onion, Pulses first)
FEATURED_CROPS = ['Potato', 'Onion', 'Pulses', 'Rice', 'Maize', 'Chilli', 'Groundnut', 'Sugarcane']

# Major States of India with Agro-Climatic Norms and Famous Mandis
LOCATION_DATA = {
    'Maharashtra': {
        'district': 'Maharashtra',
        'zone': 'Western Agro-Climatic Plateau & Ghats',
        'default_rainfall': 1050.0,
        'default_temperature': 27.5,
        'default_humidity': 64.0,
        'soil': 'Black regur soil, rich in iron, lime and magnesium',
        'notable_crops': 'Onion, Pulses (Tur/Chana), Sugarcane, Cotton, Soybean',
        'mandi': 'Lasalgaon Mandi, Nashik (Asia\'s largest Onion market) / Pune APMC'
    },
    'Uttar Pradesh': {
        'district': 'Uttar Pradesh',
        'zone': 'Upper & Middle Gangetic Plains Zone',
        'default_rainfall': 950.0,
        'default_temperature': 26.0,
        'default_humidity': 65.0,
        'soil': 'Deep, highly fertile alluvial loam',
        'notable_crops': 'Potato (India\'s largest producer), Sugarcane, Wheat, Pulses',
        'mandi': 'Agra Mandi (India\'s Potato Capital) / Kanpur Grain Mandi'
    },
    'Madhya Pradesh': {
        'district': 'Madhya Pradesh',
        'zone': 'Central Plateau & Malwa Agro-Climatic Zone',
        'default_rainfall': 1020.0,
        'default_temperature': 26.8,
        'default_humidity': 58.0,
        'soil': 'Medium to deep black soils and mixed red soils',
        'notable_crops': 'Pulses (Chana, Tur, Urad - India\'s Pulse Bowl), Soybean, Wheat',
        'mandi': 'Indore Grain & Pulse Mandi / Ujjain APMC'
    },
    'Andhra Pradesh': {
        'district': 'Andhra Pradesh',
        'zone': 'Krishna-Godavari Coastal & Southern Zone',
        'default_rainfall': 1120.0,
        'default_temperature': 28.5,
        'default_humidity': 68.0,
        'soil': 'Coastal alluvial, delta loam, red gravelly',
        'notable_crops': 'Rice/Paddy, Chilli, Groundnut, Sugarcane, Pulses',
        'mandi': 'Guntur Mirchi Yard / Vijayawada Regional APMC'
    },
    'Punjab': {
        'district': 'Punjab',
        'zone': 'Trans-Gangetic Plain Agro-Climatic Zone',
        'default_rainfall': 680.0,
        'default_temperature': 24.5,
        'default_humidity': 56.0,
        'soil': 'Alluvial sierozem soils with high canal irrigation',
        'notable_crops': 'Potato (Seed potato hub), Wheat, Rice, Maize',
        'mandi': 'Jalandhar Potato Mandi / Ludhiana Grain Market'
    },
    'Karnataka': {
        'district': 'Karnataka',
        'zone': 'Southern Transition & Dry Agro-Climatic Zone',
        'default_rainfall': 1180.0,
        'default_temperature': 26.5,
        'default_humidity': 66.0,
        'soil': 'Red sandy loam, black cotton soil in North Karnataka',
        'notable_crops': 'Maize, Pulses (Tur in Gulbarga), Onion, Groundnut',
        'mandi': 'Hubballi Agricultural Mandi / Gulbarga (Kalaburagi) Tur Mandi'
    },
    'Gujarat': {
        'district': 'Gujarat',
        'zone': 'Gujarat Plains & Saurashtra Agro-Climatic Zone',
        'default_rainfall': 820.0,
        'default_temperature': 28.2,
        'default_humidity': 60.0,
        'soil': 'Medium black, sandy loam, goradu soils',
        'notable_crops': 'Groundnut, Potato (Deesa potato belt), Onion (Mahuva), Cotton',
        'mandi': 'Rajkot APMC / Deesa Potato Market / Mahuva Onion Mandi'
    },
    'West Bengal': {
        'district': 'West Bengal',
        'zone': 'Lower Gangetic Plain Agro-Climatic Zone',
        'default_rainfall': 1480.0,
        'default_temperature': 27.0,
        'default_humidity': 75.0,
        'soil': 'Terai alluvial, fertile silt loam',
        'notable_crops': 'Potato (Hooghly, Burdwan), Rice (Paddy), Jute',
        'mandi': 'Hooghly Potato Mandi / Siliguri Regulated Market'
    },
    'Rajasthan': {
        'district': 'Rajasthan',
        'zone': 'Western Arid & Semi-Arid Agro-Climatic Zone',
        'default_rainfall': 560.0,
        'default_temperature': 28.5,
        'default_humidity': 48.0,
        'soil': 'Desert sandy, red yellow, alluvial in eastern plains',
        'notable_crops': 'Pulses (Moong, Moth, Gram), Mustard, Bajra, Maize',
        'mandi': 'Jaipur Grain Mandi / Bikaner Dal & Pulse Yard'
    },
    'Bihar': {
        'district': 'Bihar',
        'zone': 'Middle Gangetic Alluvial Plains Zone',
        'default_rainfall': 1220.0,
        'default_temperature': 26.5,
        'default_humidity': 69.0,
        'soil': 'Young alluvium, calcareous sandy loam',
        'notable_crops': 'Maize (Gulab Bagh hub), Potato, Rice, Pulses',
        'mandi': 'Gulab Bagh Mandi, Purnea (Asia\'s large maize hub) / Patna APMC'
    }
}

# Detailed Crop Agronomic Profiles & Advisory (Potato, Onion, Pulses featured)
CROP_DATA = {
    'Potato': {
        'crop': 'Potato',
        'scientific_name': 'Solanum tuberosum',
        'tagline': 'The King of Vegetables - Universal Food Security Staple',
        'category': 'Tuber / Horticultural Vegetable',
        'seasons': 'Rabi (Oct–Nov planting, Jan–March harvest)',
        'harvest_months': ['January', 'February', 'March', 'April'],
        'ideal_temp': '15°C - 24°C (Night temp <20°C for tuberization)',
        'water_req': 'Moderate (500–700 mm, regular light furrow irrigation)',
        'soil': 'Well-drained, loose sandy loam rich in organic matter (pH 5.2 - 6.4)',
        'msp': 'Market-determined / Operation Greens Price Stabilization Scheme',
        'major_districts': 'Uttar Pradesh (Agra, Farrukhabad), West Bengal (Hooghly), Punjab (Jalandhar), Bihar, Gujarat',
        'info': (
            'Potato is the third most important food crop globally and a crucial commercial cash vegetable in India. '
            'Uttar Pradesh and West Bengal alone account for nearly 55% of national production. Harvest arrivals peak '
            'between February and April, causing substantial price drops. Prices typically climb steadily from June '
            'to November as cold storage stocks deplete leading up to the festive season.'
        ),
        'selling_tips': (
            'Avoid distress selling during peak Rabi harvest (Feb–March). Store quality, blight-free tubers in registered '
            'cold storage facilities (2–4°C, 90–95% RH). Market realisations are highest between September and November '
            'prior to early Kharif tuber arrivals.'
        )
    },
    'Onion': {
        'crop': 'Onion',
        'scientific_name': 'Allium cepa',
        'tagline': 'Essential High-Value Kitchen Staple with Sharp Volatility',
        'category': 'Bulb Vegetable / Cash Commercial',
        'seasons': 'Kharif (Aug–Nov), Late Kharif (Dec–Jan), Rabi (March–May)',
        'harvest_months': ['March', 'April', 'May', 'October', 'November'],
        'ideal_temp': '18°C - 30°C',
        'water_req': 'Moderate (650–750 mm, sensitive to both drought and waterlogging)',
        'soil': 'Deep, friable loamy soil with neutral pH (6.5 - 7.5) and good drainage',
        'msp': 'Market-determined / NAFED Price Support Buffer Stock',
        'major_districts': 'Maharashtra (Nashik, Ahmednagar, Pune), Karnataka (Hubli), Gujarat (Mahuva), MP, Rajasthan',
        'info': (
            'Onion is an indispensable culinary vegetable in India with notable price volatility. Maharashtra is the undisputed '
            'production hub, with Nashik\'s Lasalgaon serving as Asia\'s largest onion trading hub. Rabi onion stores well '
            '(5–6 months in traditional ventilated Kanda Chawl), whereas Kharif onion is harvested in Oct–Nov with shorter shelf life.'
        ),
        'selling_tips': (
            'Cure harvested Rabi bulbs for 5–7 days in shade to seal the neck. Storing well-cured Rabi onions until '
            'August–October before Kharif crop arrivals can deliver a 40–70% higher price realization.'
        )
    },
    'Pulses': {
        'crop': 'Pulses',
        'scientific_name': 'Leguminosae (Cicer arietinum, Cajanus cajan, Vigna)',
        'tagline': 'Protein Pillar of India & Soil-Enriching Nitrogen Fixer',
        'category': 'Grain Legumes / Essential Food Pulses',
        'seasons': 'Kharif (Tur, Moong, Urad: June–Nov) & Rabi (Chana, Masoor: Oct–April)',
        'harvest_months': ['February', 'March', 'April', 'November', 'December'],
        'ideal_temp': '20°C - 32°C',
        'water_req': 'Low to Moderate (400–600 mm, climate-resilient rainfed)',
        'soil': 'Deep loamy to medium-heavy black regur soils with neutral pH',
        'msp': 'Tur (Arhar): ₹7,550/q | Chana: ₹5,440/q | Moong: ₹8,682/q',
        'major_districts': 'Madhya Pradesh, Maharashtra, Rajasthan, Karnataka, Uttar Pradesh, Andhra Pradesh',
        'info': (
            'Pulses are essential protein staples for the Indian population and naturally enrich soil fertility through '
            'rhizobial nitrogen fixation. India is the world\'s largest producer, consumer, and importer of pulses. '
            'Chana (Bengal Gram) dominates Rabi production, while Tur (Pigeonpea) and Moong dominate Kharif.'
        ),
        'selling_tips': (
            'Sun-dry grains below 10% moisture to guard against pulse beetle (Callosobruchus). Avail NAFED MSP procurement '
            'centers if market mandi rates dip below minimum support prices during peak harvest windows.'
        )
    },
    'Rice': {
        'crop': 'Rice',
        'scientific_name': 'Oryza sativa',
        'tagline': 'Staple Cereal of India & Global Grain Leader',
        'category': 'Cereal / Food Grain',
        'seasons': 'Kharif (June–Nov) & Rabi (Nov–April)',
        'harvest_months': ['October', 'November', 'December', 'April'],
        'ideal_temp': '22°C - 34°C',
        'water_req': 'High (Canal & tube-well irrigation)',
        'soil': 'Heavy clayey loam to alluvial soils with high water retention',
        'msp': '₹2,300 per quintal (Common) / ₹2,320 per quintal (Grade A)',
        'major_districts': 'West Bengal, Punjab, Uttar Pradesh, Andhra Pradesh, Bihar',
        'info': (
            'Rice (paddy) is India\'s staple food grain, cultivated across both river deltas and canal-irrigated plains. '
            'Major varieties include Samba Mahsuri, PR-126, Swarna, and premium Basmati in northern belts.'
        ),
        'selling_tips': (
            'Sell to FCI / State Civil Supplies Corporation at guaranteed MSP, or hold aged fine-grain rice for private mills.'
        )
    },
    'Maize': {
        'crop': 'Maize',
        'scientific_name': 'Zea mays',
        'tagline': 'High-Yield Industrial Cereal & Poultry Feed Engine',
        'category': 'Coarse Cereal / Industrial Feed',
        'seasons': 'Kharif (June–Sept) & Rabi (Oct–Feb)',
        'harvest_months': ['September', 'October', 'February', 'March'],
        'ideal_temp': '21°C - 30°C',
        'water_req': 'Moderate (500–800 mm)',
        'soil': 'Well-drained fertile loamy soils',
        'msp': '₹2,225 per quintal',
        'major_districts': 'Karnataka, Madhya Pradesh, Bihar, Punjab, Andhra Pradesh',
        'info': (
            'Maize demand has surged due to poultry feed, animal feed, starch, and bio-fuel blending. '
            'Bihar and Andhra Pradesh achieve premier yields in Rabi maize.'
        ),
        'selling_tips': (
            'Ensure grain moisture is below 12% to prevent aflatoxin rejection by poultry mills.'
        )
    },
    'Chilli': {
        'crop': 'Chilli',
        'scientific_name': 'Capsicum annuum',
        'tagline': 'The Fiery Cash Spice - Global Export Leader',
        'category': 'Spices / High-Value Cash Crop',
        'seasons': 'Rabi (Nursery Aug–Sept, Harvest Jan–April)',
        'harvest_months': ['January', 'February', 'March', 'April'],
        'ideal_temp': '20°C - 32°C',
        'water_req': 'Moderate with controlled furrow or drip irrigation',
        'soil': 'Black cotton soils and well-drained fertile red loams',
        'msp': 'Market-determined (Guntur Yard Benchmark: ₹12,000–₹22,000/quintal)',
        'major_districts': 'Andhra Pradesh (Guntur), Karnataka (Byadagi), Madhya Pradesh, Maharashtra',
        'info': (
            'India produces over 36% of global chilli supplies. Guntur and Byadagi are premier spot pricing benchmarks.'
        ),
        'selling_tips': (
            'Cold storage in Guntur or Hubli allows holding dried pods until July–October peak export windows.'
        )
    },
    'Groundnut': {
        'crop': 'Groundnut',
        'scientific_name': 'Arachis hypogaea',
        'tagline': 'High-Oilseed Cash Crop & Protein Legume',
        'category': 'Oilseed / Legume',
        'seasons': 'Kharif (June–Oct) & Rabi (Nov–March)',
        'harvest_months': ['October', 'November', 'March', 'April'],
        'ideal_temp': '24°C - 32°C',
        'water_req': 'Moderate (500–700 mm)',
        'soil': 'Well-drained sandy loam rich in calcium',
        'msp': '₹6,783 per quintal',
        'major_districts': 'Gujarat (Saurashtra), Andhra Pradesh (Rayalaseema), Karnataka, Rajasthan',
        'info': (
            'Major domestic edible oil source. Rajkot and Kurnool mandis are leading trade hubs.'
        ),
        'selling_tips': (
            'Peak selling occurs in pre-monsoon summer when domestic solvent extraction plants stock raw pods.'
        )
    },
    'Sugarcane': {
        'crop': 'Sugarcane',
        'scientific_name': 'Saccharum officinarum',
        'tagline': 'Perennial Commercial Cash & Bio-Energy Giant',
        'category': 'Commercial / Cash Crop',
        'seasons': 'Planted Dec–March (Matures in 10–12 months)',
        'harvest_months': ['December', 'January', 'February', 'March', 'April'],
        'ideal_temp': '20°C - 38°C',
        'water_req': 'Very High (1500–2500 mm)',
        'soil': 'Deep, well-drained alluvial and black loams',
        'msp': 'Fair & Remunerative Price (FRP): ₹340 per quintal (10.25% sugar recovery)',
        'major_districts': 'Uttar Pradesh, Maharashtra, Karnataka, Tamil Nadu, Andhra Pradesh',
        'info': (
            'Crucial for sugar mills and ethanol bio-fuel blending across Western UP and Maharashtra.'
        ),
        'selling_tips': (
            'Register with cooperative sugar factories for early calendar tokens to maximize FRP delivery rate.'
        )
    }
}

def generate_predictions(crop, location, rainfall, temperature, humidity, horizon=12):
    """
    Computes rolling 12-month future prices and analytical KPIs using Random Forest model.
    """
    crop_encoded = CROP_MAPPING.get(crop)
    location_encoded = LOCATION_MAPPING.get(location)

    if crop_encoded is None or location_encoded is None:
        raise ValueError(f"Invalid crop '{crop}' or location '{location}'")

    today = pd.Timestamp.today().replace(day=1)
    future_dates = pd.date_range(today, periods=horizon, freq='MS')

    # Features: Location, Crop, Rainfall, Temperature, Humidity, Month, Year
    input_df = pd.DataFrame({
        'Location': [location_encoded] * horizon,
        'Crop': [crop_encoded] * horizon,
        'Rainfall': [float(rainfall)] * horizon,
        'Temperature': [float(temperature)] * horizon,
        'Humidity': [float(humidity)] * horizon,
        'Month': [d.month for d in future_dates],
        'Year': [d.year for d in future_dates]
    })

    if random_forest_model is None:
        raise RuntimeError("Machine Learning model is not loaded.")

    predicted_prices = random_forest_model.predict(input_df)

    price_data = []
    prices_list = []
    prev_price = None

    for date, price in zip(future_dates, predicted_prices):
        price_val = round(float(price), 2)
        quintal_val = round(price_val * 100, 2)
        prices_list.append(price_val)

        month_label = date.strftime('%B %Y')
        short_month = date.strftime('%b %y')

        if prev_price is None:
            pct_change = 0.0
            trend_direction = 'neutral'
        else:
            diff = price_val - prev_price
            pct_change = round((diff / prev_price) * 100, 2)
            if pct_change > 0.5:
                trend_direction = 'up'
            elif pct_change < -0.5:
                trend_direction = 'down'
            else:
                trend_direction = 'neutral'

        price_data.append({
            'month': month_label,
            'short_month': short_month,
            'month_number': date.month,
            'year': date.year,
            'price': price_val,
            'price_quintal': quintal_val,
            'pct_change': pct_change,
            'trend': trend_direction
        })
        prev_price = price_val

    min_price = min(prices_list)
    max_price = max(prices_list)
    avg_price = round(float(np.mean(prices_list)), 2)
    avg_quintal = round(avg_price * 100, 2)

    min_item = next(item for item in price_data if item['price'] == min_price)
    max_item = next(item for item in price_data if item['price'] == max_price)

    volatility = round(((max_price - min_price) / avg_price) * 100, 1) if avg_price > 0 else 0.0
    if volatility < 12.0:
        volatility_level = 'Low Volatility (Stable)'
    elif volatility < 25.0:
        volatility_level = 'Moderate Volatility'
    else:
        volatility_level = 'High Volatility (Opportunity)'

    first_price = prices_list[0]
    last_price = prices_list[-1]
    net_growth = round(((last_price - first_price) / first_price) * 100, 1)

    kpis = {
        'min_price': min_price,
        'min_price_quintal': round(min_price * 100, 2),
        'min_month': min_item['month'],
        'max_price': max_price,
        'max_price_quintal': round(max_price * 100, 2),
        'max_month': max_item['month'],
        'avg_price': avg_price,
        'avg_quintal': avg_quintal,
        'volatility': volatility,
        'volatility_level': volatility_level,
        'net_growth': net_growth,
        'best_sell_month': max_item['month']
    }

    return price_data, kpis

# Initialize AgriChatbot
chatbot_engine = AgriChatbot(
    predict_fn=generate_predictions,
    crop_data=CROP_DATA,
    location_data=LOCATION_DATA,
    crop_mapping=CROP_MAPPING,
    location_mapping=LOCATION_MAPPING
)

# ----------------- WEB ROUTES ----------------- #

@app.route('/')
def index():
    return render_template(
        'index.html',
        crops=FEATURED_CROPS,
        locations=list(LOCATION_MAPPING.keys()),
        crop_details=CROP_DATA,
        location_details=LOCATION_DATA,
        metrics=model_metrics
    )

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return redirect(url_for('index'))

    try:
        crop = request.form.get('crop', '').strip()
        location = request.form.get('location', '').strip()
        rainfall = float(request.form.get('rainfall', 1000))
        temperature = float(request.form.get('temperature', 27))
        humidity = float(request.form.get('humidity', 65))

        if not crop or crop not in CROP_MAPPING:
            flash("Please select a valid crop from the list.", "danger")
            return redirect(url_for('index'))

        if not location or location not in LOCATION_MAPPING:
            flash("Please select a valid State of India from the list.", "danger")
            return redirect(url_for('index'))

        price_data, kpis = generate_predictions(
            crop=crop,
            location=location,
            rainfall=rainfall,
            temperature=temperature,
            humidity=humidity
        )

        crop_info = CROP_DATA.get(crop, {})
        loc_info = LOCATION_DATA.get(location, {})

        result = {
            'crop': crop,
            'location': location,
            'rainfall': rainfall,
            'temperature': temperature,
            'humidity': humidity,
            'price_data': price_data,
            'kpis': kpis
        }

        return render_template(
            'result.html',
            result=result,
            crop_info=crop_info,
            loc_info=loc_info,
            metrics=model_metrics
        )

    except Exception as e:
        flash(f"Error during prediction: {str(e)}", "danger")
        return redirect(url_for('index'))

@app.route('/chatbot')
def chatbot_page():
    return render_template(
        'chatbot.html',
        crops=FEATURED_CROPS,
        locations=list(LOCATION_MAPPING.keys()),
        crop_details=CROP_DATA,
        location_details=LOCATION_DATA
    )

@app.route('/about')
def about():
    return render_template('about.html', metrics=model_metrics, crops=CROP_DATA, locations=LOCATION_DATA)

# ----------------- REST & CHATBOT API ----------------- #

@app.route('/api/chat', methods=['POST'])
def api_chat():
    data = request.get_json(force=True, silent=True) or {}
    user_msg = data.get('message', '').strip()
    if not user_msg:
        return jsonify({'error': 'Message parameter is required'}), 400

    response = chatbot_engine.process_message(user_msg)
    return jsonify(response)

@app.route('/api/health', methods=['GET'])
def api_health():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'model_loaded': random_forest_model is not None,
        'model_type': 'RandomForestRegressor (Pan-India)'
    })

@app.route('/api/crops', methods=['GET'])
def api_crops():
    return jsonify({
        'crops': FEATURED_CROPS,
        'crop_details': CROP_DATA
    })

@app.route('/api/locations', methods=['GET'])
def api_locations():
    return jsonify({
        'locations': list(LOCATION_MAPPING.keys()),
        'location_details': LOCATION_DATA
    })

@app.route('/api/metrics', methods=['GET'])
def api_metrics():
    return jsonify(model_metrics)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({'error': 'JSON payload required'}), 400

    crop = data.get('crop')
    location = data.get('location')
    rainfall = data.get('rainfall')
    temperature = data.get('temperature')
    humidity = data.get('humidity')

    if not crop or crop not in CROP_MAPPING:
        return jsonify({'error': f"Invalid or missing crop. Supported: {FEATURED_CROPS}"}), 400

    if not location or location not in LOCATION_MAPPING:
        return jsonify({'error': f"Invalid or missing state. Supported: {list(LOCATION_MAPPING.keys())}"}), 400

    try:
        rainfall = float(rainfall) if rainfall is not None else LOCATION_DATA[location]['default_rainfall']
        temperature = float(temperature) if temperature is not None else LOCATION_DATA[location]['default_temperature']
        humidity = float(humidity) if humidity is not None else LOCATION_DATA[location]['default_humidity']

        price_data, kpis = generate_predictions(
            crop=crop,
            location=location,
            rainfall=rainfall,
            temperature=temperature,
            humidity=humidity
        )

        return jsonify({
            'success': True,
            'input': {
                'crop': crop,
                'location': location,
                'rainfall': rainfall,
                'temperature': temperature,
                'humidity': humidity
            },
            'kpis': kpis,
            'forecast': price_data,
            'crop_profile': CROP_DATA.get(crop),
            'location_profile': LOCATION_DATA.get(location)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
