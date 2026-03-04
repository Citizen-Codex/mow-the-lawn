import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

LABEL_SNAKE = "snake"
LABEL_SPIRAL = "spiral"
LABEL_RANDOM_WALK = "random_walk"
VALID_LABELS = (LABEL_SNAKE, LABEL_SPIRAL, LABEL_RANDOM_WALK)
VALID_PATH_RE = re.compile(r"^[udlr]+$")


@dataclass(frozen=True)
class EvaluationResult:
    accuracy: float
    confusion: list[list[int]]
    report: str


@dataclass(frozen=True)
class PredictionResult:
    label: str
    probabilities: dict[str, float]


def normalize_path(path: str) -> str:
    normalized = path.strip().lower()
    if not normalized:
        raise ValueError("Path cannot be empty")
    if VALID_PATH_RE.fullmatch(normalized) is None:
        raise ValueError("Path must contain only the characters: u, d, l, r")
    return normalized


def load_labeled_paths(csv_path: str | Path) -> tuple[list[str], list[str]]:
    file_path = Path(csv_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    texts: list[str] = []
    labels: list[str] = []

    with file_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"snake_path", "spiral_path", "random_walk_path"}
        if reader.fieldnames is None or not required_columns.issubset(
            reader.fieldnames
        ):
            raise ValueError(
                "CSV must include 'snake_path', 'spiral_path', and 'random_walk_path' columns"
            )

        for row in reader:
            snake_path = normalize_path(row["snake_path"])
            spiral_path = normalize_path(row["spiral_path"])
            random_walk_path = normalize_path(row["random_walk_path"])

            texts.append(snake_path)
            labels.append(LABEL_SNAKE)
            texts.append(spiral_path)
            labels.append(LABEL_SPIRAL)
            texts.append(random_walk_path)
            labels.append(LABEL_RANDOM_WALK)

    return texts, labels


def build_pipeline(random_state: int = 42) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "vectorizer",
                CountVectorizer(
                    analyzer="char",
                    ngram_range=(2, 6),
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )


def train_and_evaluate(
    csv_path: str | Path,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[Pipeline, EvaluationResult]:
    texts, labels = load_labeled_paths(csv_path)

    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    model = build_pipeline(random_state=random_state)
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions, labels=list(VALID_LABELS)).tolist()
    report = classification_report(y_test, predictions, digits=3)

    return model, EvaluationResult(accuracy=accuracy, confusion=matrix, report=report)


def classify_path(model: Pipeline, path: str) -> PredictionResult:
    normalized = normalize_path(path)
    probabilities_raw = model.predict_proba([normalized])[0]
    labels = list(model.classes_)
    probabilities = {
        label: float(prob)
        for label, prob in zip(labels, probabilities_raw, strict=True)
    }
    predicted_label = labels[int(probabilities_raw.argmax())]

    return PredictionResult(label=predicted_label, probabilities=probabilities)


def export_model(pipeline: Pipeline, output_path: str | Path) -> None:
    vectorizer: CountVectorizer = pipeline.named_steps["vectorizer"]
    clf: LogisticRegression = pipeline.named_steps["classifier"]

    payload = {
        "vocabulary": vectorizer.vocabulary_,
        "ngram_range": list(vectorizer.ngram_range),
        "coef": clf.coef_.tolist(),
        "intercept": clf.intercept_.tolist(),
        "classes": clf.classes_.tolist(),
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload), encoding="utf-8")
    print(f"Model exported to {out}")


def _run_export(csv_path: str | Path, output_path: str | Path) -> None:
    texts, labels = load_labeled_paths(csv_path)
    model = build_pipeline()
    model.fit(texts, labels)
    export_model(model, output_path)


def _run_eval(csv_path: str | Path) -> None:
    _, eval_result = train_and_evaluate(csv_path)
    print(f"Accuracy: {eval_result.accuracy:.3f}")
    print("Confusion matrix [snake, spiral, random_walk]:")
    for row in eval_result.confusion:
        print(row)
    print("Classification report:")
    print(eval_result.report)


def _run_classify(csv_path: str | Path, path: str) -> None:
    texts, labels = load_labeled_paths(csv_path)
    model = build_pipeline()
    model.fit(texts, labels)
    result = classify_path(model, path)

    print(f"Predicted strategy: {result.label}")
    print("Probabilities:")
    for label in VALID_LABELS:
        if label in result.probabilities:
            print(f"  {label}: {result.probabilities[label]:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train and use a path classifier that predicts whether a path is "
            "more similar to snake or spiral strategy"
        )
    )
    parser.add_argument(
        "--dataset",
        default="data/labelled_paths.csv",
        help="Path to labelled CSV dataset",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("eval", help="Train/evaluate on a validation split")

    export_parser = subparsers.add_parser(
        "export", help="Train on full dataset and export model as JSON"
    )
    export_parser.add_argument(
        "--output",
        default="data/model.json",
        help="Output path for the exported model JSON",
    )

    classify_parser = subparsers.add_parser(
        "classify", help="Classify one arbitrary path"
    )
    classify_parser.add_argument(
        "--path",
        required=True,
        help="Path string using only u/d/l/r characters",
    )

    args = parser.parse_args()
    dataset = Path(args.dataset)

    if args.command == "eval":
        _run_eval(dataset)
        return

    if args.command == "export":
        _run_export(dataset, args.output)
        return

    _run_classify(dataset, args.path)


if __name__ == "__main__":
    main()
