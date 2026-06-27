import os
import requests
from datetime import date
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY", "")
RAILAPI_KEY  = os.getenv("RAILAPI_KEY", "")

PNR_HOST = "irctc-indian-railway-pnr-status.p.rapidapi.com"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/check")
def check_pnr():
    pnr = request.args.get("pnr", "").strip()
    if not pnr or not pnr.isdigit() or len(pnr) != 10:
        return jsonify({"error": "Invalid PNR number."}), 400
    if not RAPIDAPI_KEY:
        return jsonify({"error": "RAPIDAPI_KEY not configured."}), 500
    try:
        headers = {
            "x-rapidapi-key": RAPIDAPI_KEY,
            "x-rapidapi-host": PNR_HOST,
            "Content-Type": "application/json",
        }
        response = requests.get(
            f"https://{PNR_HOST}/getPNRStatus/{pnr}",
            headers=headers,
            timeout=10
        )
        response.raise_for_status()
        return jsonify(response.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/trains_between")
def trains_between():
    from_code = request.args.get("from", "").strip()
    to_code   = request.args.get("to", "").strip()
    if not from_code or not to_code:
        return jsonify({"error": "from and to are required."}), 400
    if not RAILAPI_KEY:
        return jsonify({"error": "RAILAPI_KEY not configured."}), 500
    try:
        today = date.today().isoformat()
        url = f"https://api.railradar.in/v1/trains/between/{from_code}/{to_code}?date={today}"
        response = requests.get(
            url,
            headers={"Authorization": f"Bearer {RAILAPI_KEY}"},
            timeout=10
        )
        response.raise_for_status()
        return jsonify(response.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_ENV", "production") == "development"
    app.run(host="0.0.0.0", port=port, debug=debug)
