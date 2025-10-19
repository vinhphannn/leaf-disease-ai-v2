$env:MODEL_PATH = "models/model_best.pth"
$env:IMG_SIZE = "224"

Write-Host "Launching web app..."
py -m pip install gradio -q
py web/app.py


