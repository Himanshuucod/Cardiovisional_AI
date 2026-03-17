from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
import pickle
import numpy as np

app = Flask(__name__)

app.config['SECRET_KEY'] = 'cardiovision_secret'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'

db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.init_app(app)

# Load ML Model
model = pickle.load(open("model/model.pkl","rb"))
scaler = pickle.load(open("model/scaler.pkl","rb"))
accuracy = pickle.load(open("model/accuracy.pkl","rb"))

# ---------------- USER MODEL ----------------

class User(UserMixin, db.Model):

    id = db.Column(db.Integer, primary_key=True)

    username = db.Column(db.String(100), unique=True)

    password = db.Column(db.String(100))


@login_manager.user_loader
def load_user(user_id):

    return User.query.get(int(user_id))


# ---------------- ROUTES ----------------

@app.route("/")
def home():

    return redirect("/login")


# REGISTER PAGE
@app.route("/register", methods=["GET","POST"])
def register():

    if request.method == "POST":

        username = request.form["username"]
        password = request.form["password"]

        user = User(username=username,password=password)

        db.session.add(user)
        db.session.commit()

        return redirect("/login")

    return render_template("register.html")


# LOGIN PAGE
@app.route("/login", methods=["GET","POST"])
def login():

    if request.method == "POST":

        username = request.form["username"]
        password = request.form["password"]

        user = User.query.filter_by(username=username,password=password).first()

        if user:

            login_user(user)

            return redirect("/dashboard")

    return render_template("login.html")


# DASHBOARD
@app.route("/dashboard")
@login_required
def dashboard():

    return render_template("dashboard.html",accuracy=round(accuracy*100,2))


# PREDICTION PAGE
@app.route("/predict")
@login_required
def predict_page():

    return render_template("predict.html")


# ML PREDICTION
@app.route("/result",methods=["POST"])
@login_required
def result():

    age=float(request.form["age"])
    gender=float(request.form["gender"])
    height=float(request.form["height"])
    weight=float(request.form["weight"])
    ap_hi=float(request.form["ap_hi"])
    ap_lo=float(request.form["ap_lo"])
    cholesterol=float(request.form["cholesterol"])
    gluc=float(request.form["gluc"])
    smoke=float(request.form["smoke"])
    alco=float(request.form["alco"])
    active=float(request.form["active"])

    bmi=weight/((height/100)**2)

    pulse=ap_hi-ap_lo

    features=[[age,gender,height,weight,ap_hi,ap_lo,cholesterol,gluc,smoke,alco,active,bmi,pulse]]

    features_scaled=scaler.transform(features)

    prediction=model.predict(features_scaled)[0]

    if prediction==1:
        result="High Risk of Heart Disease"
    else:
        result="Low Risk"

    return render_template("result.html",prediction=result)


# LOGOUT
@app.route("/logout")
@login_required
def logout():

    logout_user()

    return redirect("/login")

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)