from flask import Flask, render_template_string, request
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load trained SVR model
with open('svm.pkl', 'rb') as f:
    model = pickle.load(f)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Student Performance Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap" rel="stylesheet">
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Poppins', sans-serif;
        }

        body {
            background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
            color: #f8fafc;
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 2rem 1rem;
        }

        .container {
            background: rgba(30, 41, 59, 0.7);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            padding: 2.5rem;
            max-width: 800px;
            width: 100%;
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5),
                        0 0 30px rgba(99, 102, 241, 0.2);
        }

        h2 {
            text-align: center;
            font-size: 2rem;
            margin-bottom: 0.5rem;
            background: linear-gradient(to right, #818cf8, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        p.subtitle {
            text-align: center;
            color: #94a3b8;
            margin-bottom: 2rem;
            font-size: 0.95rem;
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1.2rem;
        }

        .input-group {
            display: flex;
            flex-direction: column;
        }

        label {
            font-size: 0.85rem;
            margin-bottom: 0.4rem;
            color: #cbd5e1;
            font-weight: 500;
        }

        input, select {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid #334155;
            border-radius: 10px;
            padding: 0.75rem 1rem;
            color: #fff;
            font-size: 0.9rem;
            transition: all 0.3s ease;
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.3);
        }

        input:focus, select:focus {
            outline: none;
            border-color: #6366f1;
            box-shadow: 0 0 12px rgba(99, 102, 241, 0.4),
                        inset 0 2px 4px rgba(0,0,0,0.3);
        }

        button {
            grid-column: 1 / -1;
            margin-top: 1rem;
            padding: 0.9rem;
            background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            box-shadow: 0 10px 20px rgba(99, 102, 241, 0.3);
            transition: all 0.3s ease;
        }

        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 25px rgba(99, 102, 241, 0.5);
        }

        button:active {
            transform: translateY(0);
        }

        .result-box {
            margin-top: 2rem;
            padding: 1.2rem;
            border-radius: 12px;
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.3);
            text-align: center;
            box-shadow: 0 10px 25px rgba(16, 185, 129, 0.15);
            animation: fadeIn 0.5s ease-in-out;
        }

        .result-box h3 {
            color: #34d399;
            font-size: 1.4rem;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>

<div class="container">
    <h2>Performance Predictor</h2>
    <p class="subtitle">Enter student details to generate predicted score</p>

    <form method="POST" action="/predict" class="form-grid">
        <div class="input-group">
            <label>Age</label>
            <input type="number" name="age" required min="10" max="100" value="20">
        </div>

        <div class="input-group">
            <label>Gender (0: Female, 1: Male)</label>
            <input type="number" name="gender" required min="0" max="1" value="0">
        </div>

        <div class="input-group">
            <label>Course Code</label>
            <input type="number" name="course" required value="1">
        </div>

        <div class="input-group">
            <label>Study Hours / Day</label>
            <input type="number" step="0.1" name="study_hours" required value="5.5">
        </div>

        <div class="input-group">
            <label>Class Attendance (%)</label>
            <input type="number" step="0.1" name="class_attendance" required value="85.0">
        </div>

        <div class="input-group">
            <label>Internet Access (0: No, 1: Yes)</label>
            <input type="number" name="internet_access" required min="0" max="1" value="1">
        </div>

        <div class="input-group">
            <label>Sleep Hours / Night</label>
            <input type="number" step="0.1" name="sleep_hours" required value="7.0">
        </div>

        <div class="input-group">
            <label>Sleep Quality (1-5)</label>
            <input type="number" name="sleep_quality" required min="1" max="5" value="4">
        </div>

        <div class="input-group">
            <label>Study Method Code</label>
            <input type="number" name="study_method" required value="1">
        </div>

        <div class="input-group">
            <label>Facility Rating (1-5)</label>
            <input type="number" name="facility_rating" required min="1" max="5" value="3">
        </div>

        <div class="input-group">
            <label>Exam Difficulty (1-5)</label>
            <input type="number" name="exam_difficulty" required min="1" max="5" value="3">
        </div>

        <button type="submit">Predict Result</button>
    </form>

    {% if prediction_text %}
    <div class="result-box">
        <h3>{{ prediction_text }}</h3>
    </div>
    {% endif %}
</div>

</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            float(request.form['age']),
            float(request.form['gender']),
            float(request.form['course']),
            float(request.form['study_hours']),
            float(request.form['class_attendance']),
            float(request.form['internet_access']),
            float(request.form['sleep_hours']),
            float(request.form['sleep_quality']),
            float(request.form['study_method']),
            float(request.form['facility_rating']),
            float(request.form['exam_difficulty'])
        ]
        
        final_features = [np.array(features)]
        prediction = model.predict(final_features)
        output = round(prediction[0], 2)

        return render_template_string(HTML_TEMPLATE, prediction_text=f'Predicted Score: {output}')
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
