from flask import Flask, render_template, request
import pickle

tokenizer = pickle.load(open("models/cv.pkl", "rb"))
model = pickle.load(open("models/clf.pkl", "rb"))

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    email_body = request.form.get("email_content")
    tokenized_email = tokenizer.transform([email_body])
    prediction = model.predict(tokenized_email)
    if prediction == 1:
        prediction = "Not Spam"
    else:
        prediction = "Spam"
    return render_template("index.html", prediction=prediction)

if __name__=="__main__":
    app.run(debug = True)