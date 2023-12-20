import os.path
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from pydub import AudioSegment
from io import BytesIO
import jiwer
import requests


app = Flask(__name__)
app.config['ALLOWED_EXTENSIONS'] = {'wav'}
app.config['UPLOAD_FOLDER'] = "static/uploads"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Load model and define vocabulary
model = load_model('ctc_model_v1.h5', compile=False)
print("Model loaded successfully")
model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True))

characters = [x for x in "abcdefghijklmnopqrstuvwxyz'?! "]
char_to_num = keras.layers.StringLookup(vocabulary=characters, oov_token="")
num_to_char = keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)
def preprocess_audio(audio_path, frame_length=256, frame_step=160, fft_length=384):
    try:
        audio = AudioSegment.from_file(BytesIO(tf.io.read_file(audio_path).numpy()), format="wav", frame_rate=16000, channels=1, sample_width=2)

        audio_array = np.array(audio.get_array_of_samples())
        audio_tensor = tf.constant(audio_array, dtype=tf.float32)

        spectrogram = tf.signal.stft(
            audio_tensor, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length
        )
        spectrogram = tf.abs(spectrogram)
        spectrogram = tf.math.pow(spectrogram, 0.5)

        means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
        stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
        spectrogram = (spectrogram - means) / (stddevs + 1e-10)

        spectrogram = tf.expand_dims(spectrogram, axis=-1)

        print(f"Spectrogram shape: {spectrogram.shape}")

        return tf.expand_dims(spectrogram, axis=0)
    except Exception as e:
        print(f"Error preprocessing audio: {e}")
        return None

def convert_mp3_to_wav(mp3_file_path, wav_file_path):
    # Konversi dari MP3 ke WAV
    audio = AudioSegment.from_mp3(mp3_file_path)
    audio.export(wav_file_path, format="wav")
    print("Konversi ke WAV berhasil")

def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
    output_text = []
    for result in results:
        decoded_str = tf.strings.reduce_join(num_to_char(result)).numpy().decode("utf-8")
        output_text.append(decoded_str)
    return output_text

def calculate_wer(targets_list, decoded_predictions):
    wer_scores = [jiwer.wer(targets, decoded_predictions) for targets in targets_list]
    return wer_scores

@app.route("/")
def index():
    return jsonify({
        "status": {
            "code": 200,
            "message": "Success fetching the API",
        },
        "data": None
    }), 200

@app.route("/predict", methods=["POST"])
def prediction():
    if request.method == "POST":
        if 'file' not in request.files:
            return "No file part"

        file = request.files['file']

        if file.filename == '':
            return "No selected file"

        if file:
            try:
                file_path_mp3 = "temp_audio.mp3"
                file_path_wav = "temp_audio.wav"
                file.save(file_path_mp3)

                if file.filename.endswith('.mp3'):
                    convert_mp3_to_wav(file_path_mp3, file_path_wav)

                # Proses preprocessing
                spectrogram = preprocess_audio(file_path_wav)

                # Lakukan prediksi
                prediction = model.predict(spectrogram)
                decoded_predictions = decode_batch_predictions(prediction)

                # Perhitungan akurasi
                targets_list = ["your", "ground", "truth", "transcription"]  # Gantilah dengan target yang sesuai
                accuracy = calculate_accuracy(targets_list, decoded_predictions)

                return jsonify({
                    "prediction": decoded_predictions,
                    "accuracy": accuracy,
                    "status": {
                        "code": 200,
                        "message": "Success",
                    }
                })
            except Exception as e:
                return jsonify({
                    "status": {
                        "code": 500,
                        "message": f"Error during prediction: {str(e)}",
                    },
                    "prediction": None
                })


def calculate_accuracy(targets_list, decoded_predictions):
    correct_predictions = sum(
        1 for target, prediction in zip(targets_list, decoded_predictions) if target == prediction)
    total_predictions = len(decoded_predictions)

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    return accuracy

def calculate_precision(targets_list, decoded_predictions):
    true_positives = 0
    predicted_positives = 0

    for target, prediction in zip(targets_list, decoded_predictions):
        # Assuming binary classification, adjust logic accordingly for multi-class
        if prediction == 1:  # Replace 1 with the positive class label
            predicted_positives += 1
            if target == prediction:
                true_positives += 1

    precision = true_positives / predicted_positives if predicted_positives > 0 else 0
    return precision

if __name__ == "__main__":
    app.run(debug=True)


if __name__ == "__main__":
    app.run(debug=True)
