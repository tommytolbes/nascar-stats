"""
utils.py -- shared helpers for the NASCAR fantasy project
"""

from difflib import SequenceMatcher

MATCH_THRESHOLD = 0.78


def fuzzy_similarity(a, b):
    """Return a 0–1 similarity score between two strings (case-insensitive)."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def match_driver(scraped_name, all_drivers, threshold=MATCH_THRESHOLD):
    """
    Find the best matching driver for a scraped name.

    Parameters
    ----------
    scraped_name : str
        The name as it appeared in the scraped source.
    all_drivers : list of (driver_id, display_name)
        Full driver list from the drivers table.
    threshold : float
        Minimum similarity score to accept a match (default 0.78).

    Returns
    -------
    (driver_id, display_name, score)  or  None if no match found.
    """
    best_id, best_name, best_score = None, None, 0.0
    for driver_id, display_name in all_drivers:
        score = fuzzy_similarity(scraped_name, display_name)
        if score > best_score:
            best_score = score
            best_id    = driver_id
            best_name  = display_name
    if best_score >= threshold:
        return best_id, best_name, best_score
    return None
