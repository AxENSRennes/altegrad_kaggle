## Submissions overview

### Submission 4
Cette soumission correspond à la configuration suivante :
- `version_embed = "v1"`
- `version_gnn = "v4"`
- Script de soumission : `retrieval_answer_v4.py`

Elle repose sur une approche de **retrieval relativement simple**, avec un reranking léger.  
Cette solution présente un **bon compromis entre performance et complexité**, et reste cohérente avec l’architecture du modèle principal.

---

### Submission 5
Cette soumission utilise la même configuration d’embeddings et de GNN :
- `version_embed = "v1"`
- `version_gnn = "v4"`
- Script de soumission : `retrieval_answer_v5.py`

En revanche, le pipeline de retrieval est ici **nettement plus complexe**, faisant appel à un modèle de type Neural (réseau de neurones dédié au retrieval), contrairement au reranking plus simple utilisé en v4.

Cette sophistication supplémentaire permet d’obtenir **le meilleur score**, mais pour un **gain marginal de quelques points seulement**, au prix d’une complexité nettement plus élevée.

-- 

## Pistes d’amélioration pour la suite

- **Optimisation des hyperparamètres (HP)**  
  Mettre en place une recherche plus systématique (grid search, random search ou approche bayésienne) afin d’exploiter pleinement le potentiel du modèle actuel.

- **Implémentation plus fidèle des recommandations de l’énoncé**  
  Reprendre l’architecture et le pipeline en suivant de manière plus rigoureuse les suggestions formulées dans l’énoncé du challenge.  
  Cette direction est prometteuse, mais impliquerait une **refonte quasi complète du pipeline existant**, tant au niveau du modèle que du retrieval.
