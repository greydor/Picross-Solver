from flask import Flask, render_template
from solver import solver

app = Flask(__name__)

# Ensure templates are auto-reloaded
app.config["TEMPLATES_AUTO_RELOAD"] = True
# app.config["DEBUG"] = True


@app.route("/")
def index():
    return render_template("index.html")