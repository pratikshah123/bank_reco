from flask import Flask, request, render_template, jsonify
import pickle

app = Flask(__name__)

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return "Welcome to the prediction service!"

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        age = int(request.form['age'])
        salary = int(request.form['salary'])
        profession = int(request.form['profession'])
        
        # Prepare input for model prediction
        input_features = [[age, salary, profession]]
        prediction = model.predict(input_features)
        
        # For demonstration, let's assume you return the product as a string
        product_mapping = {
            0: "Fixed Deposits",
            1: "Investment",
            2: "Mutual Funds",
            3: "Loans"
        }
        
        predicted_product = product_mapping[prediction[0]]
        
        # Return the prediction
        return render_template('result.html', product=predicted_product)
    
    # If GET request, show the form
    return render_template('predict.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
