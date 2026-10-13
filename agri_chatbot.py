"""
agri_chatbot.py
===============
Intelligent Conversational Agent for Agricultural Market Intelligence,
Crop Cultivation Advisory, and On-Demand Price Forecasting.
Supports natural language queries in English and Hinglish terms.
"""

import re
from datetime import datetime

class AgriChatbot:
    def __init__(self, predict_fn, crop_data, location_data, crop_mapping, location_mapping):
        self.predict_fn = predict_fn
        self.crop_data = crop_data
        self.location_data = location_data
        self.crop_mapping = crop_mapping
        self.location_mapping = location_mapping

        # Aliases / Multilingual keywords
        self.crop_aliases = {
            'potato': 'Potato', 'potatoes': 'Potato', 'alu': 'Potato', 'aloo': 'Potato', 'batata': 'Potato',
            'onion': 'Onion', 'onions': 'Onion', 'pyaz': 'Onion', 'pyaaz': 'Onion', 'kanda': 'Onion',
            'pulse': 'Pulses', 'pulses': 'Pulses', 'dal': 'Pulses', 'daal': 'Pulses', 'chana': 'Pulses',
            'tur': 'Pulses', 'arhar': 'Pulses', 'moong': 'Pulses', 'urad': 'Pulses', 'lentil': 'Pulses', 'lentils': 'Pulses',
            'rice': 'Rice', 'paddy': 'Rice', 'chawal': 'Rice', 'dhan': 'Rice',
            'maize': 'Maize', 'corn': 'Maize', 'makka': 'Maize', 'bhutta': 'Maize',
            'chilli': 'Chilli', 'chili': 'Chilli', 'mirchi': 'Chilli', 'chillies': 'Chilli',
            'groundnut': 'Groundnut', 'peanut': 'Groundnut', 'moongphali': 'Groundnut', 'mungfali': 'Groundnut',
            'sugarcane': 'Sugarcane', 'ganna': 'Sugarcane'
        }

        self.state_aliases = {
            'maharashtra': 'Maharashtra', 'mh': 'Maharashtra', 'bombay': 'Maharashtra', 'pune': 'Maharashtra', 'nashik': 'Maharashtra',
            'uttar pradesh': 'Uttar Pradesh', 'up': 'Uttar Pradesh', 'agra': 'Uttar Pradesh', 'lucknow': 'Uttar Pradesh',
            'madhya pradesh': 'Madhya Pradesh', 'mp': 'Madhya Pradesh', 'indore': 'Madhya Pradesh', 'bhopal': 'Madhya Pradesh',
            'andhra pradesh': 'Andhra Pradesh', 'ap': 'Andhra Pradesh', 'andhra': 'Andhra Pradesh', 'guntur': 'Andhra Pradesh',
            'punjab': 'Punjab', 'pb': 'Punjab', 'ludhiana': 'Punjab',
            'karnataka': 'Karnataka', 'ka': 'Karnataka', 'bangalore': 'Karnataka', 'bengaluru': 'Karnataka',
            'gujarat': 'Gujarat', 'gj': 'Gujarat', 'ahmedabad': 'Gujarat', 'rajkot': 'Gujarat',
            'west bengal': 'West Bengal', 'wb': 'West Bengal', 'bengal': 'West Bengal', 'kolkata': 'West Bengal',
            'rajasthan': 'Rajasthan', 'rj': 'Rajasthan', 'jaipur': 'Rajasthan',
            'bihar': 'Bihar', 'br': 'Bihar', 'patna': 'Bihar'
        }

    def extract_crop(self, text):
        t = text.lower()
        # Sort keys by length descending to match phrases like 'potatoes' before 'potato'
        for alias, crop in sorted(self.crop_aliases.items(), key=lambda x: -len(x[0])):
            if re.search(r'\b' + re.escape(alias) + r'\b', t):
                return crop
        return None

    def extract_state(self, text):
        t = text.lower()
        for alias, state in sorted(self.state_aliases.items(), key=lambda x: -len(x[0])):
            if re.search(r'\b' + re.escape(alias) + r'\b', t):
                return state
        return None

    def process_message(self, user_msg):
        text = user_msg.strip().lower()
        extracted_crop = self.extract_crop(text)
        extracted_state = self.extract_state(text)

        # 1. Greetings
        if any(w in text for w in ['hello', 'hi', 'hey', 'namaste', 'pranam', 'kisan', 'agribot', 'start']):
            return {
                'reply': (
                    "🙏 **Namaste! I am Kisan Sahayak (AgriBot)**, your intelligent AI agriculture and price forecasting assistant.\n\n"
                    "I can help you with:\n"
                    "• 📈 **Predict Crop Prices:** 12-month forecasts for **Potato, Onion, Pulses**, and other major crops across Indian States.\n"
                    "• 🌟 **Best Selling Timing:** Find out which month gives maximum mandi rate.\n"
                    "• 🌾 **Farming & Disease Advisory:** Soil, sowing, pest control, and storage tips.\n"
                    "• 🏛️ **Government MSP & Schemes:** Minimum Support Prices, PM-KISAN, and e-NAM.\n\n"
                    "What crop or state would you like to explore today?"
                ),
                'suggestions': [
                    "🥔 Predict Potato in Uttar Pradesh",
                    "🧅 Best time to sell Onion in Maharashtra",
                    "🌱 Pulses MSP & sowing in Madhya Pradesh",
                    "🏛️ Tell me about PM-KISAN & PMFBY schemes"
                ]
            }

        # 2. Government Schemes (PM-KISAN, PMFBY, e-NAM)
        if any(w in text for w in ['scheme', 'pm-kisan', 'pm kisan', 'yojana', 'fasal bima', 'pmfby', 'subsidy', 'insurance', 'loan', 'kcc']):
            return {
                'reply': (
                    "🏛️ **Key Government Agricultural Schemes for Indian Farmers:**\n\n"
                    "1. **PM-KISAN (Pradhan Mantri Kisan Samman Nidhi):**\n"
                    "   • Direct income support of **₹6,000 per year** in 3 equal installments of ₹2,000 directly to farmer bank accounts.\n"
                    "   • Portal: `pmkisan.gov.in`\n\n"
                    "2. **PMFBY (Pradhan Mantri Fasal Bima Yojana):**\n"
                    "   • Comprehensive crop insurance covering post-harvest, localized calamities, drought, and pest attacks.\n"
                    "   • Premium rate: Only **2% for Kharif crops, 1.5% for Rabi crops, and 5% for commercial/horticultural crops**.\n\n"
                    "3. **e-NAM (National Agriculture Market):**\n"
                    "   • Online trading platform linking over 1,300 APMC mandis pan-India for transparent price discovery.\n\n"
                    "4. **Kisan Credit Card (KCC):**\n"
                    "   • Concessional credit at **4% effective interest rate** (with timely repayment subvention) for working capital."
                ),
                'suggestions': [
                    "🌱 Pulses MSP rates",
                    "🧅 Onion price in Maharashtra",
                    "🥔 Potato storage subsidy"
                ]
            }

        # 3. Minimum Support Price (MSP) Queries
        if any(w in text for w in ['msp', 'minimum support price', 'sarkari bhav', 'government rate', 'procurement']):
            reply = "🏛️ **Official Government Minimum Support Prices (MSP) Reference:**\n\n"
            if extracted_crop:
                info = self.crop_data.get(extracted_crop, {})
                reply += f"• **{extracted_crop}**: {info.get('msp', 'Market-determined by supply and demand')}\n\n"
                reply += f"ℹ️ *Note:* Pulses (Moong, Tur, Chana) have designated government procurement centers under NAFED/FCI."
            else:
                reply += (
                    "• **Pulses (Tur/Arhar):** ₹7,550 per quintal\n"
                    "• **Pulses (Chana/Gram):** ₹5,440 per quintal\n"
                    "• **Pulses (Moong):** ₹8,682 per quintal\n"
                    "• **Paddy (Common):** ₹2,300 per quintal\n"
                    "• **Paddy (Grade A):** ₹2,320 per quintal\n"
                    "• **Maize:** ₹2,225 per quintal\n"
                    "• **Potato & Onion:** Horticultural crops — prices determined at APMC Mandis (covered under Operation Greens / MIS price support intervention during market gluts).\n\n"
                    "Would you like a 12-month market price forecast for any of these crops?"
                )
            return {
                'reply': reply,
                'suggestions': [
                    "🌱 Predict Pulses price trend",
                    "🥔 Forecast Potato prices",
                    "🧅 Onion rate forecast"
                ]
            }

        # 4. Storage & Post-Harvest Losses
        if any(w in text for w in ['storage', 'store', 'cold storage', 'rot', 'spoilage', 'curing', 'preserve']):
            crop = extracted_crop or 'Potato'
            if crop == 'Potato':
                reply = (
                    "🥔 **Cold Storage Best Practices for Potato:**\n\n"
                    "1. **Pre-cooling & Curing:** Cure tubers at 15–20°C with 85% RH for 10–14 days to thicken skin and heal wounds.\n"
                    "2. **Storage Temperature:** Maintain storage at **2°C – 4°C** with **90–95% Relative Humidity** for table potatoes (or 8–10°C with CIPC sprout suppressant for processing varieties).\n"
                    "3. **Ventilation:** Periodic air circulation prevents CO₂ accumulation which causes internal black heart.\n"
                    "4. **Grading:** Eliminate cut, bruised, or blighted tubers prior to loading bags."
                )
            elif crop == 'Onion':
                reply = (
                    "🧅 **Onion Post-Harvest Curing & Storage Techniques:**\n\n"
                    "1. **Field Curing:** Allow harvested bulbs to dry in shade or windrows for 5–7 days until neck turns dry and thin.\n"
                    "2. **Storage Climate:** Onions require **cool, dry, well-ventilated structures** (Kanda Chawl) with **65–70% Humidity** and good cross ventilation.\n"
                    "3. **Avoid Moisture:** High humidity (>75%) triggers root sprouting and black mold (*Aspergillus niger*).\n"
                    "4. **Bottom Air Flow:** Store on raised wooden or wire-mesh platforms with 4–6 inches air clearance underneath."
                )
            elif crop == 'Pulses':
                reply = (
                    "🌱 **Pulses (Dal/Gram) Grain Storage Advisory:**\n\n"
                    "1. **Moisture Target:** Dry grains thoroughly in sun until moisture content drops below **10–12%**.\n"
                    "2. **Pulse Beetle (*Callosobruchus*) Prevention:** Mix 200g edible vegetable oil per 100kg grain, or use neem seed kernel powder (1%).\n"
                    "3. **Storage Container:** Use hermetic bags (PICS bags) or clean galvanized metal bins."
                )
            else:
                reply = f"📦 For **{crop}**, ensure harvest at proper physiological maturity, clean sun drying, and storage in moisture-proof containers."

            return {
                'reply': reply,
                'suggestions': [
                    f"📈 Best month to sell {crop}",
                    f"🌾 Sowing season for {crop}",
                    "🏛️ Operation Greens storage subsidy"
                ]
            }

        # 5. Price Prediction / Best Time to Sell Query (Crop + State or Crop Alone)
        if any(w in text for w in ['predict', 'forecast', 'price', 'bhav', 'rate', 'sell', 'cost', 'trend', 'future', 'when to sell', 'bechna']):
            target_crop = extracted_crop or 'Potato'
            target_state = extracted_state or 'Uttar Pradesh' if target_crop == 'Potato' else ('Maharashtra' if target_crop == 'Onion' else 'Madhya Pradesh')

            try:
                # Get state climate defaults
                clim = self.location_data[target_state]
                price_data, kpis = self.predict_fn(
                    crop=target_crop,
                    location=target_state,
                    rainfall=clim['default_rainfall'],
                    temperature=clim['default_temperature'],
                    humidity=clim['default_humidity']
                )

                crop_profile = self.crop_data.get(target_crop, {})
                current_p = price_data[0]['price']
                current_q = price_data[0]['price_quintal']

                reply = (
                    f"📊 **12-Month Market Forecast for {target_crop} in {target_state}**:\n\n"
                    f"• **Current Estimated Rate:** ₹{current_p:.2f} / kg (₹{current_q:.2f} / quintal)\n"
                    f"• 🌟 **Best Month to Sell (Peak Price):** **{kpis['best_sell_month']}** at projected **₹{kpis['max_price']:.2f} / kg** (₹{kpis['max_price_quintal']:.2f} / quintal)\n"
                    f"• ⚠️ **Lowest Price Period:** {kpis['min_month']} (₹{kpis['min_price']:.2f} / kg)\n"
                    f"• 💰 **12-Month Projected Average:** ₹{kpis['avg_price']:.2f} / kg (₹{kpis['avg_quintal']:.2f} / quintal)\n"
                    f"• 📊 **Market Volatility:** {kpis['volatility']}% ({kpis['volatility_level']})\n"
                    f"• 📈 **Expected Price Trend:** {'+' if kpis['net_growth'] >= 0 else ''}{kpis['net_growth']}%\n\n"
                    f"💡 **Advisory Tip:** {crop_profile.get('selling_tips', 'Consider timing your market arrival around peak months for best realization.')}"
                )

                return {
                    'reply': reply,
                    'forecast_card': {
                        'crop': target_crop,
                        'location': target_state,
                        'kpis': kpis,
                        'first_month': price_data[0]
                    },
                    'suggestions': [
                        f"🧅 Forecast Onion in Maharashtra",
                        f"🌱 Forecast Pulses in Madhya Pradesh",
                        f"🥔 Storage tips for {target_crop}",
                        f"🏛️ MSP for {target_crop}"
                    ]
                }
            except Exception as e:
                return {
                    'reply': f"⚠️ Unable to compute forecast for {target_crop} in {target_state}: {str(e)}",
                    'suggestions': ["Try Potato in Uttar Pradesh", "Try Onion in Maharashtra"]
                }

        # 6. Agronomic Cultivation & Disease Query
        if any(w in text for w in ['cultivat', 'grow', 'soil', 'season', 'fertiliz', 'disease', 'pest', 'blight', 'thrips', 'sowing', 'seed']):
            target_crop = extracted_crop or 'Potato'
            info = self.crop_data.get(target_crop, {})

            reply = (
                f"🌾 **Agronomic & Cultivation Guide for {target_crop} ({info.get('scientific_name', '')})**:\n\n"
                f"• **Tagline:** {info.get('tagline', '')}\n"
                f"• **Growing Season:** {info.get('seasons', 'Rabi / Kharif')}\n"
                f"• **Soil Requirement:** {info.get('soil', 'Well drained fertile loam')}\n"
                f"• **Ideal Climate:** Temperature {info.get('ideal_temp', '20-30°C')} with {info.get('water_req', 'Moderate irrigation')}\n"
                f"• **Major Indian States:** {info.get('major_districts', 'All major agricultural states')}\n\n"
                f"📖 **Overview:** {info.get('info', '')}\n\n"
                f"Would you like to forecast market prices for {target_crop} in any state?"
            )
            return {
                'reply': reply,
                'suggestions': [
                    f"📈 Predict {target_crop} price in Uttar Pradesh",
                    f"📦 Storage tips for {target_crop}",
                    "🧅 Onion cultivation guide"
                ]
            }

        # 7. Default Fallback
        return {
            'reply': (
                "🤖 I can help you forecast crop prices and analyze market trends across India!\n\n"
                "Try asking questions like:\n"
                "• *'What is the price of Potato in Uttar Pradesh?'*\n"
                "• *'When should I sell Onion in Maharashtra?'*\n"
                "• *'What is the MSP of Pulses?'*\n"
                "• *'How to store Potato to prevent rot?'*\n"
                "• *'Tell me about PM-KISAN scheme'*"
            ),
            'suggestions': [
                "🥔 Potato price forecast in UP",
                "🧅 Onion price in Maharashtra",
                "🌱 Pulses MSP & forecast in MP",
                "🏛️ PM-KISAN details"
            ]
        }
