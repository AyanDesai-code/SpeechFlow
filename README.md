<h1> A useful tool to help correct stuttering </h1>
<h2>Dependencies</h2>
This project is built off of <a href="https://github.com/amritkromana/disfluency_detection_from_audio">this</a> github repo which has been massively helpful</p>

please install these:
```
mkdir demo_models && cd demo_models
mkdir asr && cd asr
gdown --id 1BeT7m_5qv19Sb5yrZ2zhKu6fEprUoB9N -O config.json
gdown --id 13n8VrTFVq4jGouCDamkReHlHm_1yz20U -O pytorch_model.bin
cd ..
gdown --id 1GQIXgCSF3Usiuy5hkxgOl483RPX3f_SX -O language.pt
gdown --id 1wWrmopvvdhlBw-cL7EDyih9zn_IJu5Wr -O acoustic.pt
gdown --id 1LPchbScA_cuFx1XoNxpFCYZfGoJCfWao -O multimodal.pt
```

The following packages are needed: 
- pandas==1.5.0
- torch==1.12.1
- torchaudio==0.12.1
- transformers==4.22.2
- whisper_timestamped==1.14.4
- gdown==5.1.0

<h2>How to use</h2>

It's really simple: this stuttering-correction tool runs completely in the terminal. Create a <a href ="https://python.land/virtual-environments/virtualenv">Python virtual environment</a>, install the dependencies, and start speaking about a topic you want to discuss. The tool then runs a 2-minute conversation, analyzes your speech, and displays words that may need practice so you can repeat them. This exercise can help improve speech disorders, especially stuttering.

<h2>Future Plans</h2>

Although this only runs in a terminal, a GUI browser version is currently being developed and will be released later this month!


<h2>Frontend Prototype</h2>
A browser-based frontend is now included in `frontend/`.

Run it locally with:
```
cd frontend
python -m http.server 4173
```
Then open `http://localhost:4173` in your browser.

To integrate the frontend with the backend analysis API, run this in a second terminal:
```
python backend_api.py
```
The frontend will automatically call `http://localhost:8787/api/analyze-transcript` and show live backend results when available (otherwise it falls back to demo mode).

The UI is now redesigned for a vertical Raspberry Pi touchscreen layout (optimized around ~`480x800`) with a light blue + white visual style tuned for clarity.


<h2>Frontend Screens</h2>
Current simplified screens in this prototype:
- Live transcript screen
- Loading / analysis screen
- Final results screen (practice words + summary)
