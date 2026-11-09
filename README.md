# Raphaël

Un assistant vocal personnel dans l'esprit de Jarvis : on lui parle en français,
il répond à voix haute, et il va chercher sur internet ce qu'il ne sait pas.

**V1 : 100 % gratuite.** Le cerveau tourne sur Groq (offre gratuite), la recherche
sur Tavily (1000 crédits gratuits par mois), et la voix en local sur ta carte
graphique avec les modèles open source de Kyutai.

```
 micro ──► Kyutai STT ──► texte ──► LLM (Groq) ◄──► recherche_web (Tavily)
 (local, GPU)                          │
 haut-parleur ◄── Kyutai TTS ◄── réponse
 (local, GPU)
```

## 1. Installation

Il faut Python 3.11 ou plus récent. Dans WSL (Ubuntu) :

```bash
git clone https://github.com/D0gZys/Raphael.git ~/raphael
cd ~/raphael
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[voix,dev]"
cp .env.example .env
```

Puis ouvre `.env` (par exemple avec `nano .env`) et colle tes deux clés gratuites :

- **Groq** : crée un compte sur https://console.groq.com puis une clé dans *API Keys* → `LLM_API_KEY`.
- **Tavily** : crée un compte sur https://app.tavily.com, la clé est sur le tableau de bord → `TAVILY_API_KEY`.

## 2. Premier test : le mode texte (sans micro)

```bash
python -m raphael --texte
```

Pose-lui une question d'actualité (« Quelle est la météo à Lyon demain ? »).
Ajoute `-v` pour voir quand il lance une recherche web.

## 3. Le mode vocal

### 3a. Démarrer les serveurs vocaux Kyutai (une fois par session)

Dans deux terminaux WSL séparés :

```bash
~/raphael/serveurs-vocaux/demarrer-stt.sh   # reconnaissance vocale, port 8090 (~2,5 Go de VRAM)
~/raphael/serveurs-vocaux/demarrer-tts.sh   # synthèse vocale, port 8089 (~5,3 Go de VRAM)
```

La première fois, ça compile `moshi-server` (comme pour le test de Moshi) et
télécharge les modèles : compte une bonne demi-heure. Il faut `uv`
(`curl -LsSf https://astral.sh/uv/install.sh | sh`). Les scripts reprennent les
correctifs trouvés pour Ubuntu 26.04 et la RTX 5070 (CUDA 12.8, gcc 14...).

À deux, ils prennent presque 8 Go : c'est juste. Ferme les autres applis qui
utilisent la carte graphique.

### 3b. Lancer Raphaël

```bash
sudo apt install -y libportaudio2   # une seule fois, pour le micro et le son
cd ~/raphael && source .venv/bin/activate
python -m raphael
```

Parle normalement : quand tu marques une pause, Raphaël répond. **Mets un casque**,
sinon il risque de s'entendre parler. `python -m raphael --liste-audio` affiche les
micros disponibles, et `--peripherique N` permet d'en choisir un.

Sous Windows 11, WSL donne accès au micro et aux haut-parleurs. Si ça ne marche
pas, tu peux aussi lancer `python -m raphael` directement sous Windows : il se
connectera aux serveurs vocaux de WSL via `localhost`.

## Changer de cerveau

Tout se passe dans `.env` (exemples commentés dedans) : n'importe quel service
compatible avec l'API OpenAI fonctionne (Groq, Mistral, Gemini, Ollama en local...).

## Organisation du code

| Fichier | Rôle |
|---|---|
| `raphael/brain.py` | Le cerveau : personnalité, mémoire de la conversation, appels d'outils |
| `raphael/tools/web_search.py` | L'outil de recherche internet (Tavily) |
| `raphael/tools/__init__.py` | La boîte à outils : c'est là qu'on branchera domotique, agenda... |
| `raphael/voix/` | Micro, haut-parleur, clients des serveurs Kyutai STT et TTS |
| `serveurs-vocaux/` | Scripts et configurations des serveurs Kyutai |

Les tests se lancent avec `pytest`.

## Pistes pour la suite

- Faire parler Raphaël pendant que le LLM écrit encore (latence divisée par deux).
- Pouvoir lui couper la parole.
- Un mot de réveil (« Raphaël ! ») pour qu'il n'écoute pas en permanence.
- De nouveaux outils : domotique, agenda, mails, minuteurs...
