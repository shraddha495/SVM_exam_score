import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# Load the SVR model
with open('svm.pkl', 'rb') as f:
    model = pickle.load(f)

# Define feature order expected by the model
FEATURE_NAMES = [
    "age", "gender", "course", "study_hours", "class_attendance",
    "internet_access", "sleep_hours", "sleep_quality",
    "study_method", "facility_rating", "exam_difficulty"
]

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Academic Performance Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #311042 100%);
            --card-bg: rgba(255, 255, 255, 0.05);
            --card-border: rgba(255, 255, 255, 0.12);
            --accent-purple: #8b5cf6;
            --accent-blue: #3b82f6;
            --accent-pink: #ec4899;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --input-bg: rgba(15, 23, 42, 0.6);
            --shadow-glow: 0 20px 40px -15px rgba(139, 92, 246, 0.3);
            --shadow-card: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
        }

        body {
            background: var(--bg-gradient);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 2rem 1rem;
        }

        .container {
            width: 100%;
            max-width: 900px;
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--card-border);
            border-radius: 24px;
            padding: 2.5rem;
            box-shadow: var(--shadow-card);
            animation: fadeIn 0.8s ease-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .header {
            text-align: center;
            margin-bottom: 2.5rem;
        }

        .header h1 {
            font-size: 2.25rem;
            font-weight: 700;
            background: linear-gradient(90deg, #a78bfa, #f472b6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        .header p {
            color: var(--text-muted);
            font-size: 0.95rem;
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 1.25rem;
        }

        .input-group {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .input-group label {
            font-size: 0.85rem;
            font-weight: 500;
            color: var(--text-muted);
            text-transform: capitalize;
            letter-spacing: 0.5px;
        }

        .input-group input, .input-group select {
            background: var(--input-bg);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 0.75rem 1rem;
            color: var(--text-main);
            font-size: 0.95rem;
            outline: none;
            transition: all 0.3s ease;
        }

        .input-group input:focus, .input-group select:focus {
            border-color: var(--accent-purple);
            box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.25);
        }

        .submit-btn {
            grid-column: 1 / -1;
            margin-top: 1rem;
            padding: 1rem;
            border: none;
            border-radius: 12px;
            background: linear-gradient(90deg, var(--accent-purple), var(--accent-blue));
            color: white;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            box-shadow: var(--shadow-glow);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        .submit-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 25px 45px -10px rgba(139, 92, 246, 0.5);
        }

        .submit-btn:active {
            transform: translateY(0);
        }

        .result-card {
            margin-top: 2rem;
            padding: 1.5rem;
            border-radius: 16px;
            background: rgba(139, 92, 246, 0.1);
            border: 1px solid rgba(139, 92, 246, 0.3);
            text-align: center;
            display: none;
            animation: slideUp 0.5s ease-out;
        }

        @keyframes slideUp {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .result-card h2 {
            font-size: 1.1rem;
            color: var(--text-muted);
            margin-bottom: 0.5rem;
        }

        .result-card .score {
            font-size: 2.5rem;
            font-weight: 700;
            color: #f472b6;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>SVR Performance Predictor</h1>
            <p>Enter the student metrics below to compute predicted outcome</p>
        </div>
        <form id="predict-form" class="form-grid">
            <div class="input-group">
                <label>Age</label>
                <input type="number" name="age" step="any" required placeholder="e.g. 20">
            </div>
            <div class="input-group">
                <label>Gender (Encoded)</label>
                <input type="number" name="gender" step="any" required placeholder="e.g. 0 or 1">
            </div>
            <div class="input-group">
                <label>Course (Encoded)</label>
                <input type="number" name="course" step="any" required placeholder="e.g. 1">
            </div>
            <div class="input-group">
                <label>Study Hours</label>
                <input type="number" name="study_hours" step="any" required placeholder="e.g. 5.5">
            </div>
            <div class="input-group">
                <label>Class Attendance (%)</label>
                <input type="number" name="class_attendance" step="any" required placeholder="e.g. 85">
            </div>
            <div class="input-group">
                <label>Internet Access</label>
                <select name="internet_access" required>
                    <option value="1">Yes (1)</option>
                    <option value="0">No (0)</option>
                </select>
            </div>
            <div class="input-group">
                <label>Sleep Hours</label>
                <input type="number" name="sleep_hours" step="any" required placeholder="e.g. 7">
            </div>
            <div class="input-group">
                <label>Sleep Quality (1-5)</label>
                <input type="number" name="sleep_quality" step="any" required placeholder="e.g. 4">
            </div>
            <div class="input-group">
                <label>Study Method (Encoded)</label>
                <input type="number" name="study_method" step="any" required placeholder="e.g. 2">
            </div>
            <div class="input-group">
                <label>Facility Rating (1-5)</label>
                <input type="number" name="facility_rating" step="any" required placeholder="e.g. 3">
            </div>
            <div class="input-group">
                <label>Exam Difficulty (1-5)</label>
                <input type="number" name="exam_difficulty" step="any" required placeholder="e.g. 3">
            </div>
            <button type="submit" class="submit-btn">Generate Prediction</button>
        </form>

        <div id="result" class="result-card">
            <h2>Predicted Output Score</h2>
            <div id="score-val" class="score">--</div>
        </div>
    </div>

    <script>
        document.getElementById('predict-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);
            const data = {};
            formData.forEach((value, key) => data[key] = parseFloat(value));

            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            const result = await response.json();
            const resultCard = document.getElementById('result');
            const scoreVal = document.getElementById('score-val');

            if(result.prediction !== undefined) {
                scoreVal.innerText = result.prediction.toFixed(2);
                resultCard.style.display = 'block';
            } else {
                scoreVal.innerText = 'Error';
                resultCard.style.display = 'block';
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = [data[feature] for feature in FEATURE_NAMES]
        prediction = model.predict([features])[0]
        return jsonify({'prediction': float(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
