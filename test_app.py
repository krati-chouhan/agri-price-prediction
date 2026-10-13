import unittest
import json
from app import app, CROP_MAPPING, LOCATION_MAPPING, generate_predictions

class CropPredictionTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app
        self.client = self.app.test_client()
        self.app.config['TESTING'] = True

    def test_health_api(self):
        response = self.client.get('/api/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'healthy')
        self.assertTrue(data['model_loaded'])

    def test_crops_api_contains_requested_crops(self):
        response = self.client.get('/api/crops')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('crops', data)
        # Verify requested crops: Potato, Onion, Pulses
        self.assertIn('Potato', data['crops'])
        self.assertIn('Onion', data['crops'])
        self.assertIn('Pulses', data['crops'])

    def test_locations_api_contains_indian_states(self):
        response = self.client.get('/api/locations')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('locations', data)
        # Verify major Indian states
        self.assertIn('Maharashtra', data['locations'])
        self.assertIn('Uttar Pradesh', data['locations'])
        self.assertIn('Madhya Pradesh', data['locations'])

    def test_index_page_loads(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'AgriPrice', response.data)

    def test_chatbot_page_loads(self):
        response = self.client.get('/chatbot')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Kisan Sahayak', response.data)

    def test_about_page_loads(self):
        response = self.client.get('/about')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'System Architecture', response.data)

    def test_prediction_engine_logic_potato_up(self):
        price_data, kpis = generate_predictions(
            crop='Potato',
            location='Uttar Pradesh',
            rainfall=950,
            temperature=26,
            humidity=65,
            horizon=12
        )
        self.assertEqual(len(price_data), 12)
        self.assertIn('min_price', kpis)
        self.assertIn('max_price', kpis)
        self.assertIn('best_sell_month', kpis)
        self.assertGreater(kpis['max_price'], 0)
        self.assertGreaterEqual(kpis['max_price'], kpis['min_price'])

    def test_prediction_engine_logic_onion_maharashtra(self):
        price_data, kpis = generate_predictions(
            crop='Onion',
            location='Maharashtra',
            rainfall=1050,
            temperature=27.5,
            humidity=64,
            horizon=12
        )
        self.assertEqual(len(price_data), 12)
        self.assertGreater(kpis['max_price'], 0)

    def test_prediction_engine_logic_pulses_mp(self):
        price_data, kpis = generate_predictions(
            crop='Pulses',
            location='Madhya Pradesh',
            rainfall=1020,
            temperature=26.8,
            humidity=58,
            horizon=12
        )
        self.assertEqual(len(price_data), 12)
        self.assertGreater(kpis['max_price'], 0)

    def test_web_predict_post(self):
        response = self.client.post('/predict', data={
            'crop': 'Potato',
            'location': 'Uttar Pradesh',
            'rainfall': '950',
            'temperature': '26',
            'humidity': '65'
        }, follow_redirects=True)
        self.assertEqual(response.status_code, 200)

    def test_api_predict_json(self):
        payload = {
            'crop': 'Onion',
            'location': 'Maharashtra',
            'rainfall': 1050,
            'temperature': 27.5,
            'humidity': 64
        }
        response = self.client.post('/api/predict',
                                    data=json.dumps(payload),
                                    content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertEqual(len(data['forecast']), 12)
        self.assertIn('kpis', data)

    def test_chatbot_api(self):
        # 1. Greeting
        res1 = self.client.post('/api/chat', json={'message': 'hello'})
        self.assertEqual(res1.status_code, 200)
        self.assertIn('Namaste', res1.json['reply'])

        # 2. Price query on Potato
        res2 = self.client.post('/api/chat', json={'message': 'what will be the price of potato in uttar pradesh?'})
        self.assertEqual(res2.status_code, 200)
        self.assertIn('Potato', res2.json['reply'])
        self.assertIn('Uttar Pradesh', res2.json['reply'])

        # 3. MSP query on Pulses
        res3 = self.client.post('/api/chat', json={'message': 'what is the msp of pulses?'})
        self.assertEqual(res3.status_code, 200)
        self.assertIn('MSP', res3.json['reply'])

if __name__ == '__main__':
    unittest.main()
