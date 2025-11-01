from flask import Flask, request, jsonify
from gradio_client import Client, handle_file
import os
import requests
import json

app = Flask(__name__)

# ✅ Hugging Face Space OCR
client = Client("hoangphuc05/ocr-invoice")

# ✅ Gemini API config
GEMINI_API_KEY = "AIzaSyBz1K3fccXdbJ5mm1o80d_Yi7lzAIHJrrk"
GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"

@app.route("/ocr", methods=["POST"])
def ocr_and_analyze():
    """
    Nhận ảnh + danh sách categories → OCR → Gọi Gemini → Trả JSON gồm:
    store_name, date, total_amount, currency, categoryId
    """
    if "image" not in request.files:
        return jsonify({"error": "❌ No image uploaded"}), 400

    f = request.files["image"]
    temp_path = f"temp_{f.filename}"
    f.save(temp_path)

    # ✅ Lấy danh sách category nếu có
    categories_json = request.form.get("categories")
    categories = None
    if categories_json:
        try:
            categories = json.loads(categories_json)
        except json.JSONDecodeError:
            return jsonify({"error": "Invalid JSON format for 'categories'"}), 400

    try:
        # 1️⃣ OCR
        ocr_text = client.predict(handle_file(temp_path), api_name="/predict")
        if os.path.exists(temp_path):
            os.remove(temp_path)

        ocr_text = ocr_text.strip() if isinstance(ocr_text, str) else str(ocr_text)
        print("🧾 OCR text preview:\n", ocr_text[:300])

        # 2️⃣ Prompt: thêm hướng dẫn phân loại category
        prompt = f"""
You are an intelligent AI specialized in extracting and understanding invoice information in ANY language.
Analyze the following OCR text and return a structured JSON object with these exact fields:

{{
  "store_name": "Store or company name",
  "date": "Invoice or transaction date (format: dd/mm/yyyy or similar)",
  "total_amount": "Total payment amount",
  "currency": "Predicted currency (e.g. VND, USD, EUR, JPY)",
  "categoryId": "Best matching category ID from provided list"
}}

The available categories are:
{json.dumps(categories, indent=2) if categories else "[]"}

If none of the categories match clearly, return null for categoryId.

Here is the OCR text:
{ocr_text}

Return only valid JSON, no explanations, no markdown.
"""

        # 3️⃣ Gọi Gemini
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        response = requests.post(GEMINI_URL, json=payload)
        data = response.json()

        if "candidates" not in data:
            return jsonify({
                "error": "Gemini API returned no candidates",
                "gemini_response": data
            }), 500

        gemini_text = data["candidates"][0]["content"]["parts"][0]["text"]

        # 4️⃣ Làm sạch JSON
        cleaned = gemini_text.replace("```json", "").replace("```", "").strip()

        try:
            json_data = json.loads(cleaned)
        except json.JSONDecodeError:
            json_data = {"raw_text": cleaned}

        # ✅ 5️⃣ Trả kết quả gọn
        filtered = {
            "Note": json_data.get("store_name"),
            "TransactionDate": json_data.get("date"),
            "Amount": json_data.get("total_amount"),
            "Currency": json_data.get("currency"),
            "categoryId": json_data.get("categoryId")
        }

        return jsonify(filtered)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)
