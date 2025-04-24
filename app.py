from flask import Flask, request, jsonify
from flask_cors import CORS
import model

app = Flask(__name__)
CORS(app)

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Extract features from JSON data
        bath = data.get('bath')
        balcony = data.get('balcony')
        total_sqft_int = data.get('total_sqft_int')
        bhk = data.get('bhk')
        price_per_sqft = data.get('price_per_sqft')
        area_type = data.get('area_type')
        availability = data.get('availability')
        location = data.get('location')
        
        # Predict house price
        predicted_price = model.predict_house_price(
            bath, balcony, total_sqft_int, bhk, 
            price_per_sqft, area_type, availability, location
        )
        
        # Convert NumPy type to standard Python float
        if hasattr(predicted_price, 'item'):
            predicted_price = predicted_price.item()
        else:
            predicted_price = float(predicted_price)
        
        return jsonify({
            'status': 'success',
            'predicted_price': predicted_price
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == "__main__":
    app.run(debug=True)
