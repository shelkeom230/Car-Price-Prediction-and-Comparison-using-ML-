from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import pickle, numpy as np
import requests

app = Flask(__name__, static_folder="templates")
cors = CORS(app)
model = pickle.load(open("LinearRegressionModel.pkl", "rb"))
car = pd.read_csv("cleaned car.csv")

# set your openrouter url and api key
OPENROUTER_API_KEY = (
    "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"
)
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


@app.route("/", methods=["GET", "POST"])
def index():
    companies = sorted(car["company"].unique())
    car_model = sorted(car["name"].unique())
    year = sorted(car["year"].unique(), reverse=True)
    fuel_type = sorted(car["fuel_type"].unique())

    companies.insert(0, "Select Company")
    return render_template(
        "index.html",
        companies=companies,
        car_models=car_model,
        years=year,
        fuel_type=fuel_type,
    )

@app.route("/compare", methods=["GET", "POST"])
def compare():
    if request.method == "GET":
        return render_template("comparecars.html")

    elif request.method == "POST":
        # 1. Get form data
        car1 = {
            "brand": request.form.get("car1_brand"),
            "model": request.form.get("car1_model"),
            "year": request.form.get("car1_year"),
            "fuel": request.form.get("car1_fuel"),
        }
        car2 = {
            "brand": request.form.get("car2_brand"),
            "model": request.form.get("car2_model"),
            "year": request.form.get("car2_year"),
            "fuel": request.form.get("car2_fuel"),
        }

        # 2. Prepare prompt
        prompt = f"""
Provide a well-structured and informative paragraph for each of the two cars below.
Each paragraph should be about 10 to 15 lines and must include the following details:
- Engine type and cc
- Mileage (ARAI)
- Fuel type
- Number of seats
- Transmission
- Price in India
- Key pros and cons
Make sure each paragraph is clearly labeled as "Car 1" and "Car 2".
While returning response, dont write "HERE ARE THE 2 PARAGRAPHS", return your output without that 
At last, give your own opinion based on the data about which among the 2 cars to choose which is value for money as your AI SUGGESTION to the customers.
Car 1:
Brand: {car1['brand']}
Model: {car1['model']}
Year: {car1['year']}
Fuel Type: {car1['fuel']}

Car 2:
Brand: {car2['brand']}
Model: {car2['model']}
Year: {car2['year']}
Fuel Type: {car2['fuel']}
"""

        # 3. Call OpenRouter with LLaMA 3.3 8B
        try:
            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            }

            data = {
                "model": "meta-llama/llama-3-8b-instruct",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
            }

            response = requests.post(OPENROUTER_URL, headers=headers, json=data)
            response.raise_for_status()

            ai_reply = response.json()
            text_content = ai_reply["choices"][0]["message"]["content"]
            return jsonify({"status": "success", "data": text_content})
        except Exception as e:
            print("some error occured in LLM response",e)


@app.route("/predict", methods=["POST"])
def predict():
    company = request.form.get("company")

    car_model = request.form.get("car_models")
    year = request.form.get("year")
    fuel_type = request.form.get("fuel_type")
    driven = request.form.get("kilo_driven")

    prediction = model.predict(
        pd.DataFrame(
            columns=["name", "company", "year", "kms_driven", "fuel_type"],
            data=np.array([car_model, company, year, driven, fuel_type]).reshape(1, 5),
        )
    )
    print(prediction)

    return str(np.round(prediction[0], 2))


if __name__ == "__main__":
    app.run(debug=False)
