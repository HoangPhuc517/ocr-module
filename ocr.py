from flask import Flask, request, jsonify
from gradio_client import Client, handle_file
import os

app = Flask(__name__)

# ✅ Kết nối tới Space OCR trên Hugging Face
# (thay bằng space của bạn nếu khác)
client = Client("hoangphuc05/ocr-invoice")

@app.route("/ocr", methods=["POST"])
def ocr():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    f = request.files["image"]

    # 📂 Lưu tạm file để gửi cho Hugging Face
    temp_path = os.path.join("temp_" + f.filename)
    f.save(temp_path)

    try:
        # 🔍 Gọi Space Hugging Face để nhận text OCR
        result = client.predict(handle_file(temp_path), api_name="/predict")

        # 🧹 Xóa file tạm
        os.remove(temp_path)

        # ✅ Trả về kết quả OCR
        return jsonify({
            "message": "✅ OCR success",
            "text": result.strip()
        })

    except Exception as e:
        # ❌ Bắt lỗi nếu Space bị timeout hoặc Hugging Face lỗi
        return jsonify({
            "error": f"OCR failed: {str(e)}"
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
