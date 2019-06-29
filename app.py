print("****************************************************************")
print("Starting flask app... In main.py ...")
print("****************************************************************")

# Server entry point
from flask import render_template
from flask_cors import CORS
import connexion
import logging
import argparse, sys

# Create a custom logger
logging.getLogger().setLevel(logging.INFO)

# Create handlers
c_handler = logging.StreamHandler()
c_handler.setLevel(logging.INFO)
c_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
c_handler.setFormatter(c_format)
logging.getLogger().addHandler(c_handler)

app = connexion.App(__name__, specification_dir='swagger/', options={"swagger_ui": True})
CORS(app.app)
app.add_api('swagger.yml')

# Routes
@app.route('/')
def home():
    return render_template('home.html')


# If we're running in stand alone mode, run the application
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', help='Environment. DEV or PROD', type= str, default='PROD')
    parser.add_argument('--port', help='Port number', type= int, default= 0)
    args = parser.parse_args()

    print('args', args)
    print('sys', sys)

    logging.info("Starting application on port {}...".format(args.port))
    app.run(host='0.0.0.0', port=args.port, debug=(args.env=='DEV'))
