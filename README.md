<h1> A useful tool to help correct stuttering </h1>
<h2>Dependencies</h2>
This project is built off of <a href="https://github.com/amritkromana/disfluency_detection_from_audio">this</a> github repo which has been massively helpful</p>

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

Its really simple! An amazing stuttering correction device that runs completely in the terminal, you just need to install the dependencies to get started, you talk into the device to tell it what you want to disscuss, which commences a 2 minute discussion on said topic, afterwards the conversation is analyzed and problematic words that have been flagged as being said wrong will be displayed and you just need to repeat them, this exercise can really help solve speech disorders especially stuttering.

<h2>Future Plans</h2>

Although this only runs in a terminal, a GUI browser version is currently being developed and will be released later this month!
