from data_utils import load_data, clean_data, split_data
from prepare_features import prepare_features
from preprocessing import build_preprocessor
from modeling import train_random_forest
from evaluation import evaluate_model

def main():

    df = load_data("HOUSE/housetrain.csv")
    df = clean_data(df)

    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    preprocessor = build_preprocessor(X_train)

    model = train_random_forest(X_train, y_train, preprocessor)

    results = evaluate_model(model, X_test, y_test)

    print("Pipeline kørt")

if __name__ == "__main__":
    main()

