# Entraînement Stage 2 sur Kaggle TPU

Ce dossier contient les fichiers nécessaires pour lancer l'entraînement Stage 2 sur TPU Kaggle via la ligne de commande.

## Prérequis

1. **API Kaggle installée et configurée** :
   ```bash
   pip install kaggle
   ```
   Assurez-vous que `~/.kaggle/kaggle.json` existe avec vos identifiants.

2. **Compétition rejointe** : Vous devez avoir rejoint la compétition `molecular-graph-captioning` sur Kaggle.

## Fichiers

| Fichier | Description |
|---------|-------------|
| `kernel-metadata.json` | Configuration du kernel (TPU activé, sources, etc.) |
| `train_stage2_kaggle.py` | Script d'entraînement exécuté sur Kaggle |

## Utilisation

### 1. Lancer l'entraînement

```bash
cd /home/axel/Altegrad_kaggle/mol-caption-code/kaggle
kaggle kernels push
```

Cette commande envoie le kernel à Kaggle et déclenche son exécution.

### 2. Suivre l'avancement

```bash
# Vérifier le statut du kernel
kaggle kernels status axelmau/mol-caption-stage2

# Voir les logs en temps réel (une fois terminé)
kaggle kernels output axelmau/mol-caption-stage2 -p ./output
```

### 3. Télécharger les résultats

```bash
kaggle kernels output axelmau/mol-caption-stage2 -p ./kaggle_output
```

Les checkpoints et fichiers de soumission seront dans `./kaggle_output/`.

## Test rapide

Pour tester que tout fonctionne avant un entraînement complet, modifiez `train_stage2_kaggle.py` :

```python
# Remplacer "--mode", "full" par :
"--mode", "quick",  # 500 échantillons, 1 epoch
```

## Notes importantes

- **Quota TPU** : Kaggle offre ~20 heures/semaine de TPU gratuit
- **torch_xla** est pré-installé sur les VMs TPU Kaggle
- Les données de la compétition sont accessibles via `/kaggle/input/altegrad-2024`
- Les sorties sont sauvegardées dans `/kaggle/working/` et téléchargeables après exécution
- Internet est activé pour cloner les repos et installer les packages

## Dépannage

### Le kernel échoue immédiatement
- Vérifiez que vous avez rejoint la compétition sur Kaggle
- Vérifiez vos identifiants dans `~/.kaggle/kaggle.json`

### Erreur "quota exceeded"
- Attendez que votre quota TPU se renouvelle (hebdomadaire)
- Utilisez `--mode quick` pour des tests plus courts

### Erreur lors du clone HuggingFace
- Vérifiez que le repo `Moinada/altegrad-mol-caption` est accessible
- Le checkpoint Stage 1 doit être présent dans ce repo

## Structure des sorties

Après exécution, les fichiers suivants seront disponibles :
- `outputs/stage2_full_best.pt` : Meilleur checkpoint Stage 2
- `outputs/stage2_full_last.pt` : Dernier checkpoint
- `outputs/submission_*.csv` : Fichier de soumission (si inference activée)
