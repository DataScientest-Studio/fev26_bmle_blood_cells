Set-Location $PSScriptRoot
$env:PYTHONPATH = "src"
& "$PSScriptRoot\.venv\Scripts\python.exe" -m streamlit run src/serving/app.py --server.port 8501
