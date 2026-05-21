from flask import Flask
import subprocess
import sys
app = Flask(__name__)
from flask import request
logs=[]


@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>SpeechFlow</title>
        <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=JetBrains+Mono:wght@400;500;600&family=Space+Grotesk:wght@400;500;700&display=swap" rel="stylesheet">
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            html, body {
                height: 100%;
                width: 100%;
            }

            body {
                background: linear-gradient(135deg, #0f0f0f 0%, #1a1a1e 50%, #0d0d0f 100%);
                background-attachment: fixed;
                font-family: 'Space Grotesk', sans-serif;
                color: #e8e8e8;
                position: relative;
                overflow: hidden;
            }

            /* Animated background grid */
            body::before {
                content: '';
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background-image: 
                    linear-gradient(0deg, transparent 24%, rgba(255, 200, 100, 0.05) 25%, rgba(255, 200, 100, 0.05) 26%, transparent 27%, transparent 74%, rgba(255, 200, 100, 0.05) 75%, rgba(255, 200, 100, 0.05) 76%, transparent 77%, transparent),
                    linear-gradient(90deg, transparent 24%, rgba(255, 200, 100, 0.05) 25%, rgba(255, 200, 100, 0.05) 26%, transparent 27%, transparent 74%, rgba(255, 200, 100, 0.05) 75%, rgba(255, 200, 100, 0.05) 76%, transparent 77%, transparent);
                background-size: 50px 50px;
                pointer-events: none;
                z-index: 0;
            }

            .container {
                position: relative;
                z-index: 1;
                height: 100vh;
                display: flex;
                flex-direction: column;
                padding: 3rem;
                overflow: hidden;
            }

            .header {
                margin-bottom: 2rem;
                animation: slideInDown 0.8s cubic-bezier(0.34, 1.56, 0.64, 1);
            }

            .title {
                font-family: 'Playfair Display', serif;
                font-size: 3.5rem;
                font-weight: 900;
                letter-spacing: -2px;
                margin-bottom: 0.5rem;
                background: linear-gradient(135deg, #ffc864 0%, #ffb347 50%, #ffa347 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }

            .subtitle {
                font-size: 0.95rem;
                color: #888;
                letter-spacing: 2px;
                text-transform: uppercase;
                font-weight: 500;
            }

            .main-content {
                flex: 1;
                display: flex;
                flex-direction: column;
                gap: 1.5rem;
                overflow: hidden;
            }

            .output-wrapper {
                flex: 1;
                display: flex;
                flex-direction: column;
                min-height: 0;
                animation: fadeIn 1s ease 0.3s both;
            }

            .output-label {
                font-size: 0.85rem;
                text-transform: uppercase;
                letter-spacing: 1.5px;
                color: #666;
                font-weight: 600;
                margin-bottom: 0.8rem;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }

            .status-indicator {
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background: #888;
                animation: pulse 2s infinite;
            }

            .status-indicator.active {
                background: #ffc864;
                box-shadow: 0 0 10px rgba(255, 200, 100, 0.6);
                animation: pulse 1s infinite;
            }

            pre {
                flex: 1;
                background: rgba(30, 30, 35, 0.6);
                border: 1px solid rgba(255, 200, 100, 0.15);
                border-radius: 8px;
                padding: 1.5rem;
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.9rem;
                line-height: 1.6;
                overflow-y: auto;
                white-space: pre-wrap;
                overflow-wrap: break-word;
                color: #d4af37;
                backdrop-filter: blur(10px);
                box-shadow: inset 0 0 30px rgba(255, 200, 100, 0.02), 0 0 50px rgba(255, 200, 100, 0.05);
                transition: all 0.3s ease;
                min-height: 0;
            }

            pre:hover {
                border-color: rgba(255, 200, 100, 0.3);
                box-shadow: inset 0 0 30px rgba(255, 200, 100, 0.05), 0 0 50px rgba(255, 200, 100, 0.1);
            }

            /* Scrollbar styling */
            pre::-webkit-scrollbar {
                width: 8px;
            }

            pre::-webkit-scrollbar-track {
                background: rgba(255, 200, 100, 0.05);
                border-radius: 4px;
            }

            pre::-webkit-scrollbar-thumb {
                background: rgba(255, 200, 100, 0.3);
                border-radius: 4px;
            }

            pre::-webkit-scrollbar-thumb:hover {
                background: rgba(255, 200, 100, 0.5);
            }

            .controls {
                display: flex;
                gap: 1rem;
                align-items: center;
                animation: slideInUp 0.8s cubic-bezier(0.34, 1.56, 0.64, 1) 0.2s both;
            }

            .start-btn {
                padding: 0.9rem 2.5rem;
                background: linear-gradient(135deg, #ffc864 0%, #ffb347 100%);
                color: #0f0f0f;
                border: none;
                border-radius: 6px;
                font-family: 'Space Grotesk', sans-serif;
                font-size: 1rem;
                font-weight: 700;
                letter-spacing: 1px;
                cursor: pointer;
                text-transform: uppercase;
                transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
                box-shadow: 0 8px 25px rgba(255, 200, 100, 0.25);
                position: relative;
                overflow: hidden;
            }

            .start-btn::before {
                content: '';
                position: absolute;
                top: 50%;
                left: 50%;
                width: 0;
                height: 0;
                background: rgba(255, 255, 255, 0.3);
                border-radius: 50%;
                transform: translate(-50%, -50%);
                transition: width 0.6s, height 0.6s;
            }

            .start-btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 12px 35px rgba(255, 200, 100, 0.35);
            }

            .start-btn:hover::before {
                width: 300px;
                height: 300px;
            }

            .start-btn:active {
                transform: translateY(0);
            }

            .start-btn.loading {
                opacity: 0.7;
                pointer-events: none;
            }

            .start-btn.loading::after {
                content: '';
                position: absolute;
                width: 16px;
                height: 16px;
                margin-left: 0.5rem;
                border: 2px solid rgba(15, 15, 15, 0.3);
                border-top: 2px solid #0f0f0f;
                border-radius: 50%;
                animation: spin 0.8s linear infinite;
            }

            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }

            @keyframes slideInDown {
                from {
                    opacity: 0;
                    transform: translateY(-30px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }

            @keyframes slideInUp {
                from {
                    opacity: 0;
                    transform: translateY(30px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }

            @keyframes fadeIn {
                from {
                    opacity: 0;
                }
                to {
                    opacity: 1;
                }
            }

            @keyframes pulse {
                0%, 100% {
                    opacity: 1;
                }
                50% {
                    opacity: 0.5;
                }
            }

            /* Responsiveness */
            @media (max-width: 1024px) {
                .container {
                    padding: 2rem;
                }

                .title {
                    font-size: 2.5rem;
                }

                pre {
                    font-size: 0.85rem;
                }
            }

            @media (max-width: 768px) {
                .container {
                    padding: 1.5rem;
                }

                .title {
                    font-size: 2rem;
                }

                pre {
                    font-size: 0.8rem;
                    padding: 1rem;
                }

                .start-btn {
                    padding: 0.8rem 2rem;
                    font-size: 0.9rem;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1 class="title">SpeechFlow</h1>
                <p class="subtitle">Real-time Processing Engine</p>
            </div>

            <div class="main-content">
                <div class="output-wrapper">
                    <div class="output-label">
                        <span class="status-indicator"></span>
                        Output Stream
                    </div>
                    <pre id="output"></pre>
                </div>

                <div class="controls">
                    <button class="start-btn" id="startBtn" onclick="startProcess()">
                        Start Process
                    </button>
                </div>
            </div>
        </div>

        <script>
            async function updateLogs() {
                const response = await fetch('/logs');
                const text = await response.text();
                const output = document.getElementById('output');
                output.textContent = text;
                
                // Auto-scroll to bottom
                if (output.scrollHeight > output.clientHeight) {
                    output.scrollTop = output.scrollHeight;
                }

                // Update status indicator
                const indicator = document.querySelector('.status-indicator');
                if (text.trim().length > 0) {
                    indicator.classList.add('active');
                } else {
                    indicator.classList.remove('active');
                }
            }

            async function startProcess() {
                const btn = document.getElementById('startBtn');
                btn.classList.add('loading');
                btn.textContent = 'Processing...';

                try {
                    await fetch('/start');
                    setTimeout(() => {
                        btn.classList.remove('loading');
                        btn.textContent = 'Start Process';
                    }, 1000);
                } catch (error) {
                    console.error('Error:', error);
                    btn.classList.remove('loading');
                    btn.textContent = 'Start Process';
                }
            }

            // Initial load
            updateLogs();

            // Update logs every 2 seconds for more responsive feel
            setInterval(updateLogs, 2000);
        </script>
    </body>
    </html>
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