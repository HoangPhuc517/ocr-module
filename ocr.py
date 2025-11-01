from flask import Flask, request, jsonify
from gradio_client import Client, handle_file
import os

app = Flask(__name__)

# ✅ Kết nối tới Space OCR trên Hugging Face
# (thay bằng Space của bạn nếu khác)
client = Client("hoangphuc05/ocr-invoice")

@app.route("/ocr", methods=["POST"])
def ocr():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    f = request.files["image"]

    # 📂 Lưu file tạm để gửi cho Hugging Face
    temp_path = f"temp_{f.filename}"
    f.save(temp_path)

    try:
        # 🔍 Gọi Space Hugging Face để nhận kết quả OCR
        result = client.predict(handle_file(temp_path), api_name="/predict")

        # 🧹 Xóa file tạm sau khi xử lý
        if os.path.exists(temp_path):
            os.remove(temp_path)

        # ✅ Trả về kết quả OCR
        return jsonify({
            "message": "✅ OCR success",
            "text": result.strip() if isinstance(result, str) else str(result)
        })

    except Exception as e:
        # ❌ Bắt lỗi nếu Space bị timeout hoặc Hugging Face gặp sự cố
        return jsonify({
            "error": f"OCR failed: {str(e)}"
        }), 500


if __name__ == "__main__":
    # ⚙️ Render sẽ inject biến môi trường PORT khi chạy app
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
