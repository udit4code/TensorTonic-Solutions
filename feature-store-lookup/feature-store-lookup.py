def feature_store_lookup(feature_store, requests, defaults):
    """
    Join offline user features with online request-time features.
    """
    # Write code here
    feature_vectors = []
    for request in requests:
        user_id = request["user_id"]
        online_features = request["online_features"]
        # Use offline features if available, otherwise use defaults.
        offline_features = feature_store.get(user_id, defaults)
        # Start with offline features
        feature_vector = dict(offline_features)
        # Overlay online features
        feature_vector.update(online_features)
        feature_vectors.append(feature_vector)

    return feature_vectors