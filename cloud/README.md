# Générer des vidéos Helios sur un GPU cloud

Helios est un modèle de 14 milliards de paramètres : il faut un **GPU NVIDIA sous Linux**. Ce dossier contient deux scripts :

| Script | Rôle |
|---|---|
| `setup.sh` | Installe tout (Python 3.11, PyTorch CUDA 12.8, dépendances) et télécharge les poids |
| `generate.sh` | Génère une vidéo à partir d'un texte, d'une image ou d'une vidéo |

## 1. Choisir la machine

| Ressource | Recommandé | Minimum |
|---|---|---|
| GPU | H100 80 Go (le plus rapide) ou A100 80 Go | 24 Go de VRAM, en mode basse VRAM (beaucoup plus lent) |
| RAM | 64 Go | 64 Go si le mode basse VRAM est utilisé |
| Disque | 150 Go de volume persistant | ~90 Go par modèle téléchargé |
| Pilote | CUDA 12.8 ou plus (pilote NVIDIA ≥ 570) | idem |

Fournisseurs possibles : **RunPod** (le plus simple, décrit ci-dessous), Lambda, Vast.ai… Les tarifs changent souvent : vérifiez le prix horaire avant de lancer. La facturation court tant que la machine est allumée.

## 2. Créer la machine (exemple RunPod)

1. Sur runpod.io → **Pods** → **Deploy**, choisir un GPU **H100 80GB** ou **A100 80GB**.
2. Modèle (template) : une image **PyTorch** avec **CUDA 12.8 ou plus**.
3. **Volume disk : 150 Go** (monté sur `/workspace`, il survit à l'arrêt du pod : les poids ne sont téléchargés qu'une fois).
4. Activer **SSH** (clé publique dans les réglages du compte) si vous voulez copier des fichiers depuis votre PC.
5. Déployer, puis ouvrir le **Web Terminal** ou se connecter en SSH.

## 3. Installer

Sur la machine, cloner le fork (il contient ce dossier `cloud/` en plus du code officiel) puis lancer l'installation :

```bash
cd /workspace
git clone --depth=1 https://github.com/elmannaham/Helios.git
cd Helios
bash cloud/setup.sh            # Helios-Distilled (~81 Go)
# bash cloud/setup.sh base     # ou : mid, all, none (sans téléchargement)
```

Le script s'arrête avec un message clair si le GPU ou le pilote ne conviennent pas. Il peut être relancé sans risque après une coupure.

## 4. Générer

```bash
# Texte → vidéo (~11 s)
bash cloud/generate.sh "A red fox running through fresh snow at sunrise, cinematic, golden light"

# Image → vidéo
bash cloud/generate.sh "A towering emerald wave surges forward" --mode i2v --image example/wave.jpg

# Vidéo → vidéo
bash cloud/generate.sh "A yellow sports car speeds along a mountain road" --mode v2v --video example/car.mp4

# Vidéo d'une minute (1452 images à 24 i/s)
bash cloud/generate.sh "Aerial drone flight over a misty pine forest" --frames 1449
```

Options utiles : `--model base` (meilleure qualité, beaucoup plus lent), `--seed 7` (autre variante), `--dry-run` (affiche la commande sans rien lancer), `--help`.

Conseils de prompt :
- **en anglais**, descriptif et précis : sujet, décor, lumière, mouvement de caméra (voir `example/prompt.txt`) ;
- le nombre d'images est arrondi au multiple de 33 supérieur ;
- en i2v/v2v, l'image est redimensionnée en 640×384.

Les vidéos sont écrites dans `output_helios/helios-<modèle>/`. Récupération sur votre PC (PowerShell), en remplaçant `PORT` et `IP` par ceux affichés par RunPod dans « Connect » (SSH activé, voir étape 2) :

```powershell
scp -P PORT "root@IP:/workspace/Helios/output_helios/helios-distilled/*.mp4" .
```

(ou via le navigateur de fichiers de Jupyter si votre template RunPod en fournit un).

## 5. Arrêter la facturation

- **Stop** le pod dès que vous avez fini : le GPU n'est plus facturé, le volume `/workspace` (poids + vidéos) est conservé, avec un petit coût de stockage.
- **Terminate** supprime tout, volume compris.

## Dépannage

| Symptôme | Cause probable / solution |
|---|---|
| `nvidia-smi introuvable` | Machine sans GPU NVIDIA : changer de machine |
| `Pilote NVIDIA … < 570` | Choisir une image/machine CUDA 12.8+ |
| `CUDA out of memory` | Ajouter `--offload`, ou réduire `--frames` |
| Processus tué sans message pendant le chargement | RAM insuffisante : prendre une machine avec plus de RAM |
| Premier chunk figé en i2v/v2v | Limite connue du modèle (voir README principal, `is_skip_first_chunk`) |

Ce qui n'a **pas** pu être testé lors de la préparation (pas de GPU disponible) : une génération réelle de bout en bout. La syntaxe des scripts et la commande produite (`--dry-run`) ont été vérifiées.
