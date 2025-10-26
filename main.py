import os

# Access the environment variables
secret_key = os.environ.get("SECRET_KEY")
api_key = os.environ.get("OTHER_API_KEY")
flask_env = os.environ.get("FLASK_ENV")

# Example: set Flask config
from flask import Flask
app = Flask(__name__)
app.config['SECRET_KEY'] = secret_key
app.config['ENV'] = flask_env

import secrets
print(secrets.token_hex(32))
