from flask import Flask, render_template, request, Response, redirect
from solver import solver, download_puzzle_file, get_random_puzzle_id


app = Flask(__name__)

# Ensure templates are auto-reloaded
app.config["TEMPLATES_AUTO_RELOAD"] = True



@app.route("/index2")
def grid():
    return render_template("index2.html")



@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":
        if "puzzle_id" in request.form:
            id = request.form["puzzle_id"]
            filename = download_puzzle_file(id)

        else:
            size = request.form["puzzle_size"]
            id = get_random_puzzle_id(size)
            filename = download_puzzle_file(id)
        
        row_hints, col_hints, grid = solver(filename)
        grid = grid.tolist()
        return redirect("/index2")
        



    if request.method == "GET":
        return render_template("index.html")

if __name__ == "__app__":
    app.run(debug=True)