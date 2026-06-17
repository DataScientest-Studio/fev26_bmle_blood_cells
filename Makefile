PYTHON = python
ROOT   = .

.PHONY: help install streamlit-app streamlit-folder test lint clean

help:
	@echo "Commandes disponibles :"
	@echo "  make install          Installer les dépendances"
	@echo "  make streamlit-app    Lancer l'app de classification (image unique)"
	@echo "  make streamlit-folder Lancer l'app d'analyse par dossier"
	@echo "  make test             Lancer les tests"
	@echo "  make lint             Vérifier le code"
	@echo "  make clean            Nettoyer les fichiers temporaires"

install:
	$(PYTHON) -m pip install -r requirements/base.txt
	$(PYTHON) -m pip install -r requirements/dev.txt

streamlit-app:
	$(PYTHON) -m streamlit run src/streamlit/image_analysis.py

streamlit-folder:
	$(PYTHON) -m streamlit run src/streamlit/folder_image_analysis.py

streamlit-new:
	$(PYTHON) -m streamlit run app/streamlit/app.py

test:
	$(PYTHON) -m pytest tests/ -v

lint:
	$(PYTHON) -m flake8 src/ --max-line-length=120 --exclude=__pycache__,drafts

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
