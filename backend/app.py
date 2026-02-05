from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("backend/model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    input_df = pd.DataFrame([{
        "commodity": data["commodity"],
        "state": data["state"],
        "year": int(data["year"]),
        "month": int(data["month"])
    }])

    prediction = model.predict(input_df)[0]

    return jsonify({
        "predicted_price": float(prediction)
    })

if __name__ == "__main__":
    app.run(debug=True)


