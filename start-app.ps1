$pythonScript = {
    Set-Location -Path ".\python-api"
    python app.py
}

$nextjsScript = {
    npm run dev
}

# Start Python API in a new PowerShell window
Start-Process powershell -ArgumentList "-Command", "& {$pythonScript}"

# Start Next.js in the current window
& $nextjsScript
