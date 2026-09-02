import hashlib
import unicodedata

def get_document_digest(document, lowercase=True, collapse_whitespace=True, hash_bits=64):
    mask = (1 << hash_bits) - 1
    # Step 1 : Normalize the Text first, followed by lower-casing and collapsing of whitespaces
    assert "text" in document, f"document {document} has no text key"
    normalized_text = unicodedata.normalize("NFKC", document["text"])
    if lowercase:
        normalized_text = normalized_text.casefold()
    if collapse_whitespace:
        normalized_text = " ".join(normalized_text.split())
    # Step 2 : Compute digest 
    digest = int.from_bytes(hashlib.sha256(normalized_text.encode("utf-8")).digest(), "big") & mask
    return normalized_text, digest
    
    
    
def stable_exact_deduplication(documents, lowercase=True,
                               collapse_whitespace=True, hash_bits=64):
    """
    Returns: dictionary containing retained IDs and duplicate ownership
    """
    retained_ids = [ ]
    removed_to_retained = {}
    buckets = { }
    for document in documents:
        # Step 1 : Get the digest of the document
        normalized_text, digest = get_document_digest(document, lowercase, collapse_whitespace, hash_bits)
        owner = None  
        if digest in buckets:
            for retained_text, retained_id  in buckets[digest]:
                if retained_text == normalized_text:
                    owner = retained_id 
                    break  
        if owner is None: 
            retained_ids.append(document["id"])
            buckets.setdefault(digest, []).append((normalized_text, document["id"]))
        else:
            removed_to_retained[document["id"]] = owner
    return {
        "retained_ids" : retained_ids,
        "removed_to_retained" : removed_to_retained
    }
