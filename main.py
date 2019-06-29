from flask import Flask

print("create flask app...")
app = Flask(__name__)

@app.route('/')
def hello_world():
    return "Hello, world!"

if __name__ == "__main__":
    print("start app...")
    app.run()
