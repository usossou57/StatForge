# Test manuel de l'interface utilisateur (T10.1)

## Objectif
Valider manuellement le bon fonctionnement de la page "Upload & Dictionnaire" de l'application Streamlit.

## Prérequis
1. Application Streamlit lancée : `uv run streamlit run app/ui/streamlit_app.py`
2. Fichier de test disponible : `tests/fixtures/sample_data.csv` (ou créer un CSV simple)

## Scénario de test

### 1. Lancement de l'application
- [ ] Lancer `uv run streamlit run app/ui/streamlit_app.py`
- [ ] Vérifier que le navigateur s'ouvre sur `http://localhost:8501`
- [ ] Vérifier que la sidebar affiche "📊 StatForge" et les 3 pages de navigation

### 2. Page "Upload & Dictionnaire" (page par défaut)
- [ ] Vérifier que le titre " Upload de fichier et dictionnaire de données" s'affiche
- [ ] Vérifier que le widget d'upload accepte les fichiers CSV, XLSX, SAV
- [ ] Uploader un fichier CSV de test (ex: 100 lignes, 5 colonnes)

### 3. Après upload réussi
- [ ] Vérifier le message de succès : "✅ Fichier chargé avec succès : X lignes, Y colonnes"
- [ ] Vérifier que l'aperçu des 10 premières lignes s'affiche correctement
- [ ] Vérifier que le dictionnaire de données est généré automatiquement
- [ ] Vérifier que le tableau des variables affiche : Nom, Type, Type sémantique, Modalités, Valeurs manquantes
- [ ] Vérifier que les statistiques descriptives s'affichent (variables numériques et catégorielles)

### 4. Navigation entre pages
- [ ] Cliquer sur " Analyse" dans la sidebar
- [ ] Vérifier que la page affiche un message d'information (placeholder T10.2)
- [ ] Cliquer sur "📝 Rapport" dans la sidebar
- [ ] Vérifier que la page affiche un message d'information (placeholder T10.3)
- [ ] Revenir à "📁 Upload & Dictionnaire"
- [ ] Vérifier que les données sont toujours en session (pas besoin de re-uploader)

### 5. Gestion des erreurs
- [ ] Uploader un fichier avec un format non supporté (ex: .txt)
- [ ] Vérifier que le message d'erreur s'affiche : "❌ Erreur lors de la lecture du fichier"
- [ ] Uploader un fichier CSV mal formé (ex: colonnes manquantes)
- [ ] Vérifier que l'application ne plante pas et affiche un message d'erreur approprié

### 6. Contrainte T8 (Confidentialité)
- [ ] Ouvrir les outils de développement du navigateur (F12)
- [ ] Aller dans l'onglet "Network" (Réseau)
- [ ] Uploader un fichier
- [ ] Vérifier qu'aucune requête HTTP n'est envoyée vers un serveur externe (LLM)
- [ ] Vérifier que le dictionnaire de données est généré localement

## Critères de succès
- Tous les points ci-dessus sont cochés ✅
- L'application ne plante pas pendant les tests
- Les données restent en session lors de la navigation entre pages
- Aucun envoi de données brutes vers un service externe

## Résultats
- Date du test : ___________
- Testeur : ___________
- Statut : ✅ PASS / ❌ FAIL
- Commentaires : ___________
