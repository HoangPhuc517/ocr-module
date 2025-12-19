from flask import Flask, request, jsonify
from gradio_client import Client, handle_file
import os
import requests
import json
from datetime import datetime
from datetime import timezone

# ✅ Tự động load file .env nếu có
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = Flask(__name__)

# ✅ Hugging Face Space OCR
client = Client("hoangphuc05/ocr-invoice")

# ✅ Gemini API config
GEMINI_API_KEY_VOICE = os.environ.get("GEMINI_API_KEY_VOICE")
GEMINI_API_KEY_OCR = os.environ.get("GEMINI_API_KEY_OCR")
GEMINI_API_KEY_EMAIL = os.environ.get("GEMINI_API_KEY_EMAIL")

GEMINI_MODEL = os.environ.get("MODEL_AI", "gemini-2.5-flash-lite")
GEMINI_VERSION = os.environ.get("GEMINI_VERSION", "v1")


# ✅ Function tạo Url
def get_gemini_url(api_key):
    """
    Hàm này nhận vào API Key và trả về URL hoàn chỉnh của Gemini.
    """
    if not api_key:
        print("⚠️ Cảnh báo: API Key đang bị rỗng!")
        return None
        
    return f"https://generativelanguage.googleapis.com/{GEMINI_VERSION}/models/{GEMINI_MODEL}:generateContent?key={api_key}"



@app.route("/ocr", methods=["POST"])
def ocr_and_analyze():
    
    print("🔔 New /ocr request received")
    print("/n" * 5)

    url_ocr = get_gemini_url(GEMINI_API_KEY_OCR)

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

                # 2️⃣ Prompt: thêm hướng dẫn phân loại category + quy tắc tiền Việt
        prompt = f"""
You are an intelligent AI specialized in extracting and understanding invoice information in ANY language.

Analyze the following OCR text and return a structured JSON object with these exact fields:

{{
  "store_name": "Store or company name",
  "date": "Invoice or transaction date (format: dd/mm/yyyy or similar)",
  "total_amount": "Total payment amount",
  "currency": "Predicted currency (e.g. VND, USD, EUR, JPY)",
  "categoryId": "Best matching category ID from provided list",
  "needRescan": "true or false depending on extraction reliability"
}}

Rules for needRescan:
- needRescan = true if total_amount is missing, unreadable, null, empty, or uncertain.
- needRescan = false if total_amount is extracted confidently.
- Do NOT rely on image quality; only judge based on OCR text content.

The available categories are:
{json.dumps(categories, indent=2) if categories else "[]"}

### Special Instruction for Currency Interpretation ###

1. **GENERAL RULE (For USD, EUR, JPY, etc.):**
   - The **dot (.)** is the decimal separator (e.g., $1,234.50).
   - The **comma (,)** is the thousand separator.
   - Example: For USD, "1,580.00" means 1580.00.

2. **SPECIFIC RULE (For VND - Vietnamese Dong):**
   - VND amounts are **ALWAYS INTEGERS** for extraction purposes.
   - For VND, **both dot (.) and comma (,) are thousand separators.**
   - If you detect VND, any trailing separators followed by two or three digits (like ".00" or ",000") should be ignored or treated as part of the integer amount.
   - **VND Example:**
     - "1.580.000" means 1580000 VND.
     - **"1,580.00" means 1580 VND.**

If none of the categories match clearly, return null for categoryId.

Here is the OCR text:
{ocr_text}

Return ONLY valid JSON. No explanations. No markdown.
"""


        # 3️⃣ Gọi Gemini
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        response = requests.post(url_ocr, json=payload)
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
            "CategoryId": json_data.get("categoryId"),
            "NeedRescan": json_data.get("needRescan")
        }

        return jsonify(filtered)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

    # ================================================================
# 2) NEW API — Classify Expenses (như C# ClassifyExpensesAsync)
# ================================================================
@app.route("/classify-expense", methods=["POST"])
def classify_expenses():
    """
    Input:
    {
        "prompt": "hôm nay đi siêu thị mua đồ 150k",
        "emotion": "vui vẻ",
        "categories": [
            { "Id": "guid...", "Name": "Ăn uống" },
            { "Id": "guid...", "Name": "Mua sắm" }
        ]
    }
    """

    url_voice = get_gemini_url(GEMINI_API_KEY_VOICE)

    try:
        data = request.get_json()

        prompt = data.get("prompt")
        emotion = data.get("emotion")
        categories = data.get("categories", [])

        if not prompt:
            return jsonify({"error": "prompt is required"}), 400

        # ===== Mapping categories =====
        category_mapping = "\n".join([f"- {c['Name']} (ID: {c['Id']})" for c in categories])

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # ===== Build instruction (copy từ C# sang Python) =====
        instruction = f"""
Bạn là chuyên gia phân tích ngôn ngữ tiếng Việt.

Dưới đây là danh sách category:
{category_mapping}

LƯU Ý QUAN TRỌNG:
- KHÔNG tự tạo record nếu thiếu số tiền hoặc thiếu category.
- Nếu không xác định được → trả về:
  detail = [], total = 0, advice = "..."
- Nếu lời nói không liên quan chi tiêu → trả về detail = [], total = 0
- Không được bịa thông tin.

Ngày hiện tại: {now}

Người dùng nói:
{prompt}

Emotion: {emotion}

Trả về JSON theo schema:
{{
  "total": 0,
  "detail": [
    {{
      "category": {{ "id": "UUID", "name": "Tên" }},
      "date": "YYYY-MM-DD HH:mm:ss",
      "price": 0,
      "note": "string"
    }}
  ],
  "advice": "string"
}}
"""

        # ===== Gemini Call Payload =====
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": instruction}]
                }
            ]
        }

        response = requests.post(url_voice, json=payload)
        result = response.json()

        if "candidates" not in result:
            return jsonify({"error": "Gemini returned no output", "raw": result}), 500

        text = result["candidates"][0]["content"]["parts"][0]["text"]
        text = text.replace("```json", "").replace("```", "").strip()

        try:
            json_data = json.loads(text)
        except:
            json_data = {"raw_text": text}

        return jsonify(json_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

# 3️⃣ [MỚI] API Phân loại Email (Port từ C# sang)
@app.route("/classify-email", methods=["POST"])
def classify_email():

    current_date = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    """
    Input JSON:
    {
        "subject": "Tiêu đề email",
        "snippet": "Đoạn trích dẫn...",
        "body": "Nội dung đầy đủ...",
        "categories": [ {"Id": "...", "Name": "..."} ]
    }
    """
    url_email = get_gemini_url(GEMINI_API_KEY_EMAIL)

    try:
        data = request.get_json()
        subject = data.get("subject", "")
        snippet = data.get("snippet", "")
        body = data.get("body", "")
        categories = data.get("categories", [])

        # 1. Xây dựng Prompt (Dịch từ C#)
        instruction = f"""Bạn là chuyên gia phân loại email. Nhiệm vụ của bạn là xác định xem email có phải là hóa đơn (invoice), biên lai (receipt), hay thông báo thanh toán không.

Các dấu hiệu email là hóa đơn/biên lai:
- Tiêu đề chứa từ khóa: hóa đơn, invoice, receipt, biên lai, thanh toán, payment, order, đơn hàng
- Nội dung chứa thông tin: số tiền, tổng tiền, total, amount, giá trị, VAT, thuế
- Có thông tin về giao dịch mua bán, thanh toán
- Có mã đơn hàng, mã giao dịch
- Đến từ các nhà cung cấp dịch vụ, cửa hàng, siêu thị, ứng dụng thanh toán

Ngày hiện tại (UTC) là: {current_date}. Nếu không xác định được ngày giao dịch trong email, hãy dùng ngày hiện tại (UTC).

Trả về JSON với format:
{{
  "isInvoice": true/false,
  "confidence": 0.0-1.0 (độ tin cậy),
  "reason": "Lý do phân loại",
  "amount": number (số tiền nếu tìm thấy, nếu không để null),
  "note": "ghi chú ngắn gọn về giao dịch (nếu có)",
  "categoryId": "GUID của category nếu map được từ danh sách category cung cấp",
  "transactionDate": "Ngày giao dịch (ISO 8601), nếu không có thì trả null"
}}"""

        if categories:
            cat_lines = "\n".join([
            f"- {c.get('Name', c.get('name', 'Unknown'))} (ID: {c.get('Id', c.get('id', 'Unknown'))})" 
            for c in categories
        ])
            instruction += f"\n\nDanh sách category khả dụng (name - ID):\n{cat_lines}\nHãy chọn đúng ID từ danh sách này nếu xác định được."

        body_preview = body[:1000] + "..." if len(body) > 1000 else body
        email_content = f"Tiêu đề: {subject}\n\nTóm tắt: {snippet}\n\nNội dung: {body_preview}"
        
        final_prompt = f"{instruction}\n\n{email_content}"

        # 2. Cấu hình JSON Schema (Giống hệt C#)
        payload = {
            "contents": [{
                "role": "user",
                "parts": [{"text": final_prompt}]
            }],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": {
                    "type": "object",
                    "properties": {
                        "isInvoice": {"type": "boolean"},
                        "confidence": {"type": "number"},
                        "reason": {"type": "string"},
                        "amount": {"type": "number"},
                        "note": {"type": "string"},
                        "categoryId": {"type": "string"},
                        "transactionDate": {"type": "string", "format": "date-time"}
                    },
                    "required": ["isInvoice", "confidence", "reason"]
                }
            }
        }

        # 3. Gọi Gemini
        response = requests.post(url_email, json=payload)
        
        if response.status_code != 200:
            print(f"❌ Gemini Error: {response.text}")
            return jsonify({"error": "Gemini API Error", "details": response.text}), response.status_code

        result = response.json()
        
        # 4. Parse kết quả
        try:
            text = result["candidates"][0]["content"]["parts"][0]["text"]
            # Gemini trả về JSON chuẩn rồi, load trực tiếp
            return jsonify(json.loads(text))
        except Exception as ex:
            # Fallback nếu lỗi parse
            return jsonify({
                "isInvoice": False,
                "confidence": 0.0,
                "reason": "Lỗi phân tích output từ AI",
                "raw": str(result)
            })

    except Exception as e:
        print(f"🔥 Exception: {str(e)}")
        return jsonify({"error": str(e)}), 500
    

# Thêm đoạn này để cron-job ping vào không bị lỗi 404
@app.route("/", methods=["GET"])
def keep_alive():
    print("🔔 Ping received at home.")
    print ("--------------------------" * 3)
    return "AI MODULE By VINANCE!", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)
