Heart Disease Prediction + Lifestyle Recommendations (Flask)
===========================================================

A simple Flask web app that estimates heart disease risk using a synthetic-trained model and provides personalized lifestyle tips.

Features
--------
- Logistic Regression model trained on synthetic data at startup
- Web form to input clinical and lifestyle data
- Probability estimate with clear High/Low risk label
- Actionable lifestyle recommendations

Tech Stack
---------
- Python, Flask
- scikit-learn, numpy
- HTML/CSS (no JS required)

Setup
-----
1. Create and activate a virtual environment (recommended).

   Windows PowerShell:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

3. Run the app:
   ```powershell
   python app.py
   ```

4. Open in browser: `http://127.0.0.1:5000/`

Notes
-----
- The model is trained on synthetic data for demonstration only. It is not a medical device and does not provide medical advice.
- For production, replace with a validated model and add proper input validation, logging, and security hardening.

Project Structure
-----------------
```
app.py
requirements.txt
static/
  styles.css
templates/
  index.html
  result.html
README.md
```


