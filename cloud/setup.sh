#!/usr/bin/env bash
# Prépare une machine Linux + GPU NVIDIA pour générer des vidéos avec Helios.
#
# Usage : bash cloud/setup.sh [distilled|base|mid|all|none]   (défaut : distilled)
#   - installe les paquets système, uv, Python 3.11 et les dépendances d'inférence
#   - télécharge les poids choisis dans le cache Hugging Face
#   - vérifie que PyTorch voit le GPU
# Le script est relançable sans risque : chaque étape déjà faite est réutilisée.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_CHOICE="${1:-distilled}"
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
MIN_DRIVER_MAJOR=570  # minimum requis par les roues PyTorch CUDA 12.8
PYTHON_VERSION="3.11"
# Paquets de requirements.txt inutiles pour l'inférence (entraînement / suivi) ou installés à part (torch).
EXCLUDED_PACKAGES='^(torch|torchvision|torchaudio|triton|deepspeed|mpi4py|wandb|hf-doc-builder|tf_keras|nvitop)(==|$)'

# Sur RunPod, /workspace est le volume persistant : les ~80 Go de poids y survivent à l'arrêt du pod.
if [[ -z "${HF_HOME:-}" && -d /workspace ]]; then
  export HF_HOME=/workspace/hf_cache
fi

log() { printf '\n==> %s\n' "$*"; }
die() { printf '\nERREUR : %s\n' "$*" >&2; exit 1; }

check_gpu() {
  log "Vérification du GPU"
  command -v nvidia-smi >/dev/null || die "nvidia-smi introuvable : cette machine n'a pas de GPU NVIDIA utilisable."
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  local driver_major
  driver_major="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1 | cut -d. -f1)"
  ((driver_major >= MIN_DRIVER_MAJOR)) \
    || die "Pilote NVIDIA ${driver_major} < ${MIN_DRIVER_MAJOR}. Choisissez une machine/image avec CUDA 12.8 ou plus."
}

install_system_packages() {
  if ! command -v apt-get >/dev/null; then
    log "apt-get absent : installez vous-même git, curl, ffmpeg, libgl1 et libglib2.0-0"
    return
  fi
  log "Installation des paquets système"
  local sudo=""
  [[ "$(id -u)" -ne 0 ]] && sudo="sudo"
  $sudo apt-get update -qq
  DEBIAN_FRONTEND=noninteractive $sudo apt-get install -y -qq git curl ffmpeg libgl1 libglib2.0-0 >/dev/null
}

install_uv() {
  export PATH="$HOME/.local/bin:$PATH"
  command -v uv >/dev/null && return
  log "Installation de uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
}

install_python_env() {
  cd "$REPO_DIR"
  if [[ ! -x .venv/bin/python ]]; then
    log "Création de l'environnement Python ${PYTHON_VERSION}"
    uv venv --python "$PYTHON_VERSION" .venv
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate

  log "Installation de PyTorch (CUDA 12.8)"
  local torch_pins
  torch_pins="$(grep -E '^(torch|torchvision|torchaudio)==' requirements.txt | tr '\n' ' ')"
  # shellcheck disable=SC2086
  uv pip install $torch_pins --index-url "$TORCH_INDEX_URL"

  log "Installation des dépendances d'inférence"
  local filtered
  filtered="$(mktemp)"
  grep -vE "$EXCLUDED_PACKAGES" requirements.txt >"$filtered"
  uv pip install -r "$filtered"
  rm -f "$filtered"
}

download_weights() {
  local repos=()
  case "$MODEL_CHOICE" in
    distilled) repos=(Helios-Distilled) ;;
    base) repos=(Helios-Base) ;;
    mid) repos=(Helios-Mid) ;;
    all) repos=(Helios-Distilled Helios-Base Helios-Mid) ;;
    none) return ;;
    *) die "Modèle inconnu « ${MODEL_CHOICE} ». Choix : distilled, base, mid, all, none." ;;
  esac
  for repo in "${repos[@]}"; do
    # transformer_ode (~57 Go) ne sert qu'à l'entraînement.
    log "Téléchargement de BestWishYsh/${repo} (~81 Go) dans ${HF_HOME:-~/.cache/huggingface}"
    hf download "BestWishYsh/${repo}" --exclude "transformer_ode/*"
  done
}

sanity_check() {
  log "Contrôle final"
  cd "$REPO_DIR"
  python - <<'EOF'
import torch

assert torch.cuda.is_available(), "PyTorch ne voit pas le GPU (CUDA indisponible)."
print(f"torch {torch.__version__} | CUDA {torch.version.cuda} | {torch.cuda.get_device_name(0)}")

from helios.diffusers_version.pipeline_helios_diffusers import HeliosPipeline  # noqa: F401

print("Import du pipeline Helios : OK")
EOF
}

check_gpu
install_system_packages
install_uv
install_python_env
download_weights
sanity_check

log "Prêt. Exemple : bash cloud/generate.sh \"A red fox running through fresh snow at sunrise\""
