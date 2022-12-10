from flask import Flask, render_template
app = Flask(__name__)


@app.route('/')
@app.route("/home")
def home():
    return render_template("table.html")


@app.route("/graph")
def graph():
    return render_template("positives.html")


if __name__ == '__main__':
    app.run(debug=True, port=8080)
