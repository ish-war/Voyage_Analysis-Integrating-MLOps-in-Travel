from flask import Flask, request, render_template
import numpy as np
import pickle
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load the trained model and scaler
model = pickle.load(open("rf_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Feature order used during training (Update based on your training set)
feature_order = [
    "from_Florianopolis (SC)", "from_Sao_Paulo (SP)", "from_Salvador (BH)", "from_Brasilia (DF)", "from_Rio_de_Janeiro (RJ)", "from_Campo_Grande (MS)",
    "from_Aracaju (SE)", "from_Natal (RN)", "from_Recife (PE)", 
    "destination_Florianopolis (SC)", "destination_Sao_Paulo (SP)", "destination_Salvador (BH)", "destination_Brasilia (DF)", "destination_Rio_de_Janeiro (RJ)", "destination_Campo_Grande (MS)",
    "destination_Aracaju (SE)", "destination_Natal (RN)", "destination_Recife (PE)", 
    "flightType_economic", "flightType_firstClass", "flightType_premium", 
    "agency_Rainbow", "agency_CloudFy", "agency_FlyingDrops", 
    "month", "year", "day"
]

@app.route("/")
def home():
    return render_template("index.html", prediction=None)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get user input
        Departure = request.form["from"]
        Destination = request.form["destination"]
        FlightType = request.form["flightType"]
        Agency = request.form["agency"]
        month = int(request.form["month"])
        year = int(request.form["year"])
        day = int(request.form["day"])

        # Create a feature vector with correct ordering
        input_features = np.zeros(len(feature_order))  # Initialize all to 0

        # Encode categorical variables (Set index positions to 1 where applicable)
        if f"from_{Departure}" in feature_order:
            input_features[feature_order.index(f"from_{Departure}")] = 1
        if f"destination_{Destination}" in feature_order:
            input_features[feature_order.index(f"destination_{Destination}")] = 1
        if f"flightType_{FlightType}" in feature_order:
            input_features[feature_order.index(f"flightType_{FlightType}")] = 1
        if f"agency_{Agency}" in feature_order:
            input_features[feature_order.index(f"agency_{Agency}")] = 1
        
        # Assign numerical values (Month, Year, Day)
        input_features[feature_order.index("month")] = month
        input_features[feature_order.index("year")] = year
        input_features[feature_order.index("day")] = day

        # Scale input features
        input_scaled = scaler.transform([input_features])  # Ensure correct transformation

        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        # Generate visualization (Price Trend Graph)
        plt.figure(figsize=(5, 3))
        plt.bar(["Predicted Price"], [prediction], color="blue")
        plt.ylabel("Price ($)")
        plt.title("Flight Price Prediction")
        plt.grid(axis="y")

        # Convert plot to image
        img = io.BytesIO()
        plt.savefig(img, format="png")
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()

        return render_template("index.html", prediction=round(prediction, 2), plot_url=plot_url)

    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
