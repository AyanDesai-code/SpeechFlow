from flask import Flask
import subprocess
import sys
app = Flask(__name__)
from flask import request
logs=[]


@app.route('/')
def index():
    return '''
    <button onclick="fetch('/start')">start</button>
    <pre id="output"></pre>

    <script>
    async function updateLogs() {
        const response = await fetch('/logs');
        const text = await response.text();

        document.getElementById('output').textContent = text;
    }

    setInterval(updateLogs, 500);
    </script>
    
    '''

@app.route('/logs')
def get_logs():
    return "\n".join(logs)



@app.route('/log', methods=['POST'])
def log():
    data = request.json
    logs.append(data['message'])

    return '', 204

@app.route('/start')
def start():
    subprocess.Popen([sys.executable, 'conversation_engine.py'])
    return '', 204

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)