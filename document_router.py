import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv("documents.csv")

texts = data["text"]
labels = data["category"]

vectorizer = TfidfVectorizer(max_features=3000, stop_words="english")
features = vectorizer.fit_transform(texts)

X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42
)

tree = DecisionTreeClassifier(max_depth=20, random_state=42)
tree.fit(X_train, y_train)
tree_preds = tree.predict(X_test)
print("Decision Tree accuracy:", round(accuracy_score(y_test, tree_preds), 4))

forest = RandomForestClassifier(n_estimators=200, random_state=42)
forest.fit(X_train, y_train)
forest_preds = forest.predict(X_test)
print("Random Forest accuracy:", round(accuracy_score(y_test, forest_preds), 4))

print("\nRandom Forest report:")
print(classification_report(y_test, forest_preds))

def route_documents(model, X, threshold=0.6):
    probabilities = model.predict_proba(X)
    routed = []
    flagged = []

    for i, probs in enumerate(probabilities):
        top_confidence = max(probs)
        predicted = model.classes_[probs.argmax()]

        if top_confidence >= threshold:
            routed.append((i, predicted))
        else:
            flagged.append((i, predicted, round(top_confidence, 2)))

    return routed, flagged

routed, flagged = route_documents(forest, X_test)
print("\nAuto routed:", len(routed))
print("Flagged for human review:", len(flagged))
