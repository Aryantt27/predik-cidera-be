from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import pandas as pd
import os

# Inisialisasi Flask
app = Flask(__name__)
CORS(app)


# Path ke file model dan scaler
MODEL_PATH = os.path.join(os.getcwd(), "model.pkl")
SCALER_PATH = os.path.join(os.getcwd(), "scaler.pkl")

# Load model
try:
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print("Model loaded successfully.")
    else:
        print(f"Error: File {MODEL_PATH} tidak ditemukan.")
        model = None
except Exception as e:
    print(f"Error loading model: {e}")
    model = None


# Load scaler
try:
    if os.path.exists(SCALER_PATH):
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        print("Scaler loaded successfully.")
    else:
        print(f"Error: File {SCALER_PATH} tidak ditemukan.")
        scaler = None
except Exception as e:
    print(f"Error loading scaler: {e}")
    scaler = None

@app.route("/")
def home():
    """
    Rute untuk halaman utama.
    """
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """
    Rute untuk prediksi cedera pemain.
    """
    # Validasi apakah model dan scaler telah dimuat
    if scaler is None:
        return jsonify({"error": "Scaler tidak dimuat. Periksa file scaler.pkl."}), 500
    if model is None:
        return jsonify({"error": "Model tidak dimuat. Periksa file model.pkl."}), 500

    try:
        # Ambil data dari request JSON
        data = request.json

        # Validasi input
        required_fields = [
            "player_age", "player_weight", "player_height",
            "previous_injuries", "training_intensity", "recovery_time"
        ]
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({"error": f"Missing fields: {', '.join(missing_fields)}"}), 400

        # Konversi data ke DataFrame
        input_data = pd.DataFrame([[
            data["player_age"],
            data["player_weight"],
            data["player_height"],
            data["previous_injuries"],
            data["training_intensity"],
            data["recovery_time"]
        ]], columns=required_fields)

        # Normalisasi data menggunakan scaler
        input_scaled = scaler.transform(input_data)

        # Prediksi menggunakan model
        prediction = model.predict(input_scaled)
        result = "Berisiko Cedera" if prediction[0] == 1 else "Tidak Berisiko Cedera"

        # Kembalikan hasil sebagai JSON
        return jsonify({"prediction": result}), 200

    except Exception as e:
        return jsonify({"error": f"Terjadi kesalahan: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
