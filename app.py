from flask import Flask
import subprocess
import sys
app = Flask(__name__)

@app.route('/')
def index():
    return '''
    <button onclick="fetch('/start')">start</button>
    '''

@app.route('/start')
def start():
    subprocess.Popen([sys.executable, 'conversation_engine.py'])
    return '', 204

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)