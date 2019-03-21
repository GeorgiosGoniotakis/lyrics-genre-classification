import pickle


def store_report(path, content):
    with open(file=path, mode="w", encoding='utf8') as f:
        f.write(content)


def store_model(path, content):
    """Exports and stores a vectorizer to a pickle file.

        Args:
            path: The output file path
            content: The vectorizer or model that needs to be stored
        """
    with open(path, "wb") as f:
        pickle.dump(content, f)
