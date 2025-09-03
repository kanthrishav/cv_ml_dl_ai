# RAG based Chat-bot

## Overview

<img width="1917" height="981" alt="image" src="https://github.com/user-attachments/assets/fc099b69-2493-4331-8b80-e2b2cba86c69" />



https://github.com/user-attachments/assets/1fb88318-4097-4944-8b43-df97fe9744c7



## Run

In your project root

      ├─ docker-compose.yml
      ├─ api/
      │  ├─ Dockerfile
      │  ├─ requirements.txt
      │  └─ main.py
      └─ ui/
         ├─ Dockerfile
         ├─ requirements.txt
         ├─ app.py
         ├─ templates/
         │  └─ index.html
         └─ static/
            ├─ main.js
            └─ style.css
      
      docker compose down -v --remove-orphans
      docker compose build --no-cache
      docker compose up -d

Check if it working or not 
      
      curl -s http://localhost:9150/health | jq
      
      Output : { "status": "ok", "docs": 0, "chunks": 0 }

Open the UI : http://<your-host>:9151



