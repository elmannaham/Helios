#!/usr/bin/env bash
# Génère une vidéo avec Helios (à lancer après cloud/setup.sh).
#
# Usage : bash cloud/generate.sh "PROMPT" [options]
#   --mode t2v|i2v|v2v        texte, image ou vidéo vers vidéo (défaut : t2v)
#   --image CHEMIN            image de départ (obligatoire en i2v)
#   --video CHEMIN            vidéo de départ (obligatoire en v2v)
#   --frames N                nombre d'images, arrondi au multiple de 33 (défaut : 240 ≈ 11 s à 24 i/s)
#   --model distilled|base|mid  (défaut : distilled, le plus rapide)
#   --seed N                  graine aléatoire (défaut : 42)
#   --offload / --no-offload  force ou désactive le mode basse VRAM (défaut : auto selon le GPU)
#   --compile                 torch.compile : plus rapide ensuite, mais longue compilation au 1er lancement
#   --dry-run                 affiche la commande sans la lancer
# Les vidéos sont écrites dans output_helios/<modèle>/.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# En dessous, les ~40 Go de poids bf16 + activations risquent de ne pas tenir : on décharge en RAM.
OFFLOAD_THRESHOLD_MIB=70000

die() { printf 'ERREUR : %s\n' "$*" >&2; exit 1; }
usage() { sed -n '2,14p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

[[ $# -ge 1 ]] || usage 1
[[ "$1" == "-h" || "$1" == "--help" ]] && usage 0
PROMPT="$1"
shift

MODE="t2v" IMAGE="" VIDEO="" FRAMES=240 MODEL="distilled" SEED=42 OFFLOAD="auto" COMPILE=0 DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="${2:?}"; shift 2 ;;
    --image) IMAGE="${2:?}"; shift 2 ;;
    --video) VIDEO="${2:?}"; shift 2 ;;
    --frames) FRAMES="${2:?}"; shift 2 ;;
    --model) MODEL="${2:?}"; shift 2 ;;
    --seed) SEED="${2:?}"; shift 2 ;;
    --offload) OFFLOAD=1; shift ;;
    --no-offload) OFFLOAD=0; shift ;;
    --compile) COMPILE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h | --help) usage 0 ;;
    *) die "Option inconnue : $1 (voir --help)" ;;
  esac
done

[[ -n "$PROMPT" ]] || die "Le prompt est vide."
[[ "$FRAMES" =~ ^[0-9]+$ && "$SEED" =~ ^[0-9]+$ ]] || die "--frames et --seed doivent être des entiers positifs."

# Réglages repris de scripts/inference/helios-<modèle>_t2v.sh
case "$MODEL" in
  distilled) REPO="BestWishYsh/Helios-Distilled"
    PRESET=(--guidance_scale 1.0 --is_enable_stage2 --pyramid_num_inference_steps_list 2 2 2 --is_amplify_first_chunk) ;;
  base) REPO="BestWishYsh/Helios-Base"
    PRESET=(--guidance_scale 5.0) ;;
  mid) REPO="BestWishYsh/Helios-Mid"
    PRESET=(--guidance_scale 5.0 --is_enable_stage2 --pyramid_num_inference_steps_list 20 20 20 --use_zero_init --zero_steps 1) ;;
  *) die "Modèle inconnu « $MODEL ». Choix : distilled, base, mid." ;;
esac

case "$MODE" in
  t2v) INPUT=() ;;
  i2v) [[ -f "$IMAGE" ]] || die "Mode i2v : --image doit pointer vers un fichier existant."
    INPUT=(--image_path "$(realpath "$IMAGE")") ;;
  v2v) [[ -f "$VIDEO" ]] || die "Mode v2v : --video doit pointer vers un fichier existant."
    INPUT=(--video_path "$(realpath "$VIDEO")") ;;
  *) die "Mode inconnu « $MODE ». Choix : t2v, i2v, v2v." ;;
esac

if [[ "$OFFLOAD" == "auto" ]]; then
  OFFLOAD=0
  if command -v nvidia-smi >/dev/null; then
    vram_mib="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
    ((vram_mib < OFFLOAD_THRESHOLD_MIB)) && OFFLOAD=1
  fi
fi

OPTIONS=()
if ((OFFLOAD)); then
  ((COMPILE)) && die "--compile et le mode basse VRAM ne sont pas compatibles."
  OPTIONS+=(--enable_low_vram_mode --group_offloading_type leaf_level)
fi
((COMPILE)) && OPTIONS+=(--enable_compile)

CMD=(python infer_helios.py
  --base_model_path "$REPO" --transformer_path "$REPO"
  --sample_type "$MODE" --prompt "$PROMPT" --num_frames "$FRAMES" --seed "$SEED"
  "${INPUT[@]}" "${PRESET[@]}" "${OPTIONS[@]}"
  --output_folder "./output_helios/helios-$MODEL")

cd "$REPO_DIR"
if ((DRY_RUN)); then
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

[[ -x .venv/bin/python ]] || die "Environnement absent : lancez d'abord « bash cloud/setup.sh »."
# shellcheck disable=SC1091
source .venv/bin/activate
[[ -z "${HF_HOME:-}" && -d /workspace/hf_cache ]] && export HF_HOME=/workspace/hf_cache

((OFFLOAD)) && echo "Mode basse VRAM activé (déchargement en RAM) : plus lent, prévoir 64 Go de RAM."
"${CMD[@]}"
echo "Vidéo enregistrée dans $REPO_DIR/output_helios/helios-$MODEL/ :"
ls -t "output_helios/helios-$MODEL" | head -n1
