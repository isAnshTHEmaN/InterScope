import os

from flask import Flask, request, jsonify
from dotenv import load_dotenv
from agent import agent
import json
load_dotenv()

app = Flask(__name__)

@app.route("/generateIdeas", methods=["POST"])
def generateIdeas():
    # print("Received request to generate ideas")
    request_data = request.get_json()
    topicOne = request_data.get("topicOne", "")
    topicTwo = request_data.get("topicTwo", "")
    timeFrame = request_data.get("timeFrame", 12)
    # print("Generating ideas for:", topicOne, topicTwo, timeFrame)
    ideas = json.loads(agent(topicOne, topicTwo, timeFrame))
    # ideas = {}

    # print(ideas)
    return jsonify(ideas)
#curl -X POST http://104.198.68.208:5000/generateIdeas -H "Content-Type: application/json" -d '{"topicOne": "AI", "topicTwo": "Healthcare", "timeFrame": "2025-01-01"}'
# On Windows, use the following command:
# Invoke-RestMethod -Uri "http://104.198.68.208:5000/generateIdeas" -Method POST -ContentType "application/json" -Body '{"topicOne": "AI", "topicTwo": "Healthcare", "timeFrame": "2025-01-01"}'
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
    #MUST KNOW COMMANDS
    # gunicorn --workers 4 --bind 0.0.0.0:8080 main:app --daemon
    # pkill -f gunicorn