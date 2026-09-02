import hashlib
import unicodedata

# NFKC (Normalization Form Compatibility Composition) is a Unicode normalization scheme that converts different Unicode representations that are intended to be treated equivalently into a more consistent representation. 
# For example, compatibility characters such as "①" can normalize to "1", and "ﬁ" (a single ligature character) can normalize to "fi". We use unicodedata.normalize("NFKC", text) before deduplication because two documents may look or mean effectively the same but contain different underlying Unicode code points; without normalization, their byte sequences—and therefore SHA-256 hashes—would differ. 
# In short: NFKC reduces superficial Unicode differences so your exact-deduplication operates on a more canonical representation of the text.

def get_document_digest(document, lowercase=True, collapse_whitespace=True, hash_bits=64):
    # GOAL : We use the hash only as a candidate finder : 
    # text -> normalize via NFKC -> lowercase -> collapse-whitepsaces -> SHA-256 -> 64 bit digest -> key for the bucket
    mask = (1 << hash_bits) - 1
    # Step 1 : Normalize the Text first, followed by lower-casing and collapsing of whitespaces
    assert "text" in document, f"document {document} has no text key"
    # Conceptually, we do : raw unicode text -> normalize equivalent/compatibility Unicode forms -> canonoicalized text.
    # This matters because two strings can look effectively identical while having different underlying Unicode code points.
    normalized_text = unicodedata.normalize("NFKC", document["text"])
    if lowercase:
        # casefold() is stronger and more Unicode-aware than simply calling .lower(), which makes it useful for text comparison.
        normalized_text = normalized_text.casefold()
    if collapse_whitespace:
        normalized_text = " ".join(normalized_text.split())
    # Step 2 : Compute digest 
    # At first, we convert the string to bytes via normalized_text.encode("utf-8")). 
    # Conceptually, we convert "hello world!" to b"hello world!". 
    # After that, we compute the SHA-256 via hashlib.sha256(...).digest(). 
    # SHA-256 takes an arbitrary-length sequence of bytes and produces exactly 256 bits = 32 bytes. 
    # The important property is that the same normalized text deterministically produces the same SHA-256 hash.
    # After that, we convert those 32 bytes into an integer via int.from_bytes(..., "big") 
    # We're interpreting those 256 bits as one large integer between 0 and 2^256. 
    # But, we keep only hash_bits bits. By default, hash_bits = 64. 
    # So, mask = (1 << 64) - 1 , which creates 111111…111​ (64 ones). 
    # Then, digest = full_hash_integer & mask, which keeps only the lowest 64 bits of the 256 bit SHA-256 value. 
    # Why truncate SHA-256 to 64 bits? 
    # Mostly space and lookup efficiency. We don't need the full cryptographic 256-bit value just to decide which bucket to search.​
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
        # Step 2 : If document digest is found in the bucket, then, find the first retained_text that matches with the normalized text of the current document
        if digest in buckets:
            for retained_text, retained_id  in buckets[digest]:
                if retained_text == normalized_text:
                    owner = retained_id 
                    break  
        if owner is None: 
            # It means, that the current document has no matching digest and hence, we need to create a new bucket exclusively with the new digest as the search key. 
            retained_ids.append(document["id"])
            buckets.setdefault(digest, []).append((normalized_text, document["id"]))
        else:
            # Otherwise, we can safely say that current document has a matching document with same digest and hence, we can set its parent/owner to the already added matching document. 
            removed_to_retained[document["id"]] = owner
    return {
        "retained_ids" : retained_ids,
        "removed_to_retained" : removed_to_retained
    }


# Stable exact document deduplication is commonly done during dataset/data-pipeline preprocessing, especially before training language models, building search indexes, or creating RAG corpora. For example, if we're preparing 100 million web pages for LLM pretraining, the same article may have been crawled multiple times or appear under different URLs. 
# We normalize each document and deduplicate it so the model doesn't repeatedly train on identical content. Similarly, before indexing documents into Elasticsearch/vector databases for search or RAG, exact dedup prevents storing and retrieving redundant copies.

# The stable part means that when duplicates occur, we deterministically keep the first occurrence and map later duplicates back to it: doc_17 → retained, doc_92 → doc_17, doc_501 → doc_17. This is useful in reproducible ETL/ML pipelines because running the same ordered dataset through the pipeline gives the same retained documents and preserves provenance — we know exactly which document survived and which documents were removed as its duplicates.

