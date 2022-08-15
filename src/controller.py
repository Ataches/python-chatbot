from flask import Flask
from flask import request
from models import Message
import chatgui
import json

app = Flask(__name__)


@app.route("/", methods=['POST'])
def post_method():
    return json.dumps(chatgui.start(str(Message(json.loads(request.data)).content))), 200


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)