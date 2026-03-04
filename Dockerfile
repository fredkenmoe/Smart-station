# 1. Utiliser une image Python légère
FROM python:3.10-slim

# 2. Créer un dossier de travail dans le serveur
WORKDIR /app

# 3. Copier le fichier des dépendances
COPY requirements.txt .

# 4. Installer les bibliothèques
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copier tout ton code source dans le serveur
COPY . .

# 6. Lancer l'interface Streamlit
EXPOSE 8501
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
