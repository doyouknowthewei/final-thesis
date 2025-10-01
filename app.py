from flask import Flask, render_template

app = Flask(__name__, template_folder="templates")  # Explicitly set templates folder

@app.route("/")
def home():
    return render_template("html.html")  # Ensure the filename matches exactly

if __name__ == "__main__":
    app.run(debug=True, port=5000)
