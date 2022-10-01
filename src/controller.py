from flask import Flask, jsonify
from flask import request
from models import Message
import chatgui
import json

from src.serializers import MessageSerializer

app = Flask(__name__)


@app.route("/", methods=['POST'])
def post_method():
    response: Message = chatgui.start(str(Message(json.loads(request.data)).content))
    return jsonify(MessageSerializer.serialise(response))


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)