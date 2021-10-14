import os

from app import create_app
from flask import jsonify
from flask_swagger import swagger
from flask_swagger_ui import get_swaggerui_blueprint

settings_module = os.getenv('APP_SETTINGS_MODULE')
app = create_app(settings_module)


@app.route("/spec")
def spec():
    return jsonify(swagger(app))

### swagger specific ###
SWAGGER_URL = '/swagger'
API_URL = '/spec'
SWAGGERUI_BLUEPRINT = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "Single-Cell API REST"
    }
)

app.register_blueprint(SWAGGERUI_BLUEPRINT, url_prefix=SWAGGER_URL)

