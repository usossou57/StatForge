"""Modèles de données pour la représentation interne des datasets.

Ce module définit les structures de données utilisées pour représenter
les variables et les dictionnaires de données dans StatLab.
"""

import json
from dataclasses import asdict, dataclass, field


@dataclass
class Variable:
    """Représentation interne d'une variable d'un dataset.

    Attributes:
        name: Nom de la variable dans le dataset.
        label: Label descriptif de la variable (optionnel).
        dtype: Type de données détecté ("numeric", "categorical", "datetime", "text").
        semantic_type: Type sémantique ("binary", "ordinal", "count", "continuous").
        levels: Liste des niveaux pour les variables catégorielles.
        labels: Dictionnaire de correspondance entre codes et labels.
        missing_values: Liste des valeurs considérées comme manquantes.
        role: Rôle de la variable dans l'analyse ("outcome", "predictor", "id", "weight").
        reference_category: Catégorie de référence pour les variables catégorielles.
        weight: Nom de la variable de pondération associée.
        metadata: Dictionnaire de métadonnées supplémentaires.
    """

    name: str
    label: str | None = None
    dtype: str = "unknown"
    semantic_type: str | None = None
    levels: list[str] | None = None
    labels: dict[str, str] | None = None
    missing_values: list = field(default_factory=list)
    role: str | None = None
    reference_category: str | None = None
    weight: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class DataDictionary:
    """Dictionnaire de données interne d'un dataset importé.

    Contient les métadonnées complètes d'un dataset, incluant toutes les
    variables et leurs caractéristiques.

    Attributes:
        dataset_name: Nom du dataset.
        n_rows: Nombre de lignes (observations) dans le dataset.
        n_columns: Nombre de colonnes (variables) dans le dataset.
        variables: Dictionnaire des variables, indexé par nom de variable.
        source_path: Chemin vers le fichier source du dataset.
        file_hash: Hash du fichier source pour vérification d'intégrité.
    """

    dataset_name: str
    n_rows: int
    n_columns: int
    variables: dict[str, Variable]
    source_path: str
    file_hash: str

    def to_json(self) -> str:
        """Sérialise le dictionnaire de données en JSON.

        Convertit l'objet DataDictionary et toutes ses variables en une
        chaîne JSON valide, prête à être stockée ou transmise.

        Returns:
            Chaîne JSON représentant le dictionnaire de données.

        Example:
            >>> data_dict = DataDictionary(...)
            >>> json_str = data_dict.to_json()
            >>> print(json_str)
            '{"dataset_name": "...", "n_rows": 100, ...}'
        """
        data = asdict(self)
        return json.dumps(data, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "DataDictionary":
        """Désérialise un JSON en DataDictionary.

        Reconstruit un objet DataDictionary complet à partir d'une chaîne JSON,
        incluant la reconstruction de tous les objets Variable.

        Args:
            json_str: Chaîne JSON représentant un dictionnaire de données.

        Returns:
            Instance de DataDictionary reconstruite depuis le JSON.

        Raises:
            json.JSONDecodeError: Si la chaîne JSON est invalide.
            KeyError: Si des champs requis sont manquants dans le JSON.

        Example:
            >>> json_str = '{"dataset_name": "...", ...}'
            >>> data_dict = DataDictionary.from_json(json_str)
            >>> print(data_dict.dataset_name)
        """
        data = json.loads(json_str)

        # Reconstruire les objets Variable à partir du dictionnaire
        variables = {}
        for var_name, var_data in data["variables"].items():
            variables[var_name] = Variable(**var_data)

        # Créer l'instance DataDictionary avec les variables reconstruites
        return cls(
            dataset_name=data["dataset_name"],
            n_rows=data["n_rows"],
            n_columns=data["n_columns"],
            variables=variables,
            source_path=data["source_path"],
            file_hash=data["file_hash"],
        )
