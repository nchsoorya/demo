from difflib import SequenceMatcher

try:
    import streamlit as st
except Exception:
    class _DummyStreamlit:
        @staticmethod
        def cache_resource(func):
            return func

    st = _DummyStreamlit()

SIMILARITY_THRESHOLD = 0.75


@st.cache_resource
def load_local_embedding_model():
    """Optional hook kept for compatibility; returns None in Catalyst mode."""
    return None


def clean_education_duplicates_semantically(edu_history: list) -> list:
    n = len(edu_history)
    if n <= 1:
        return edu_history

    school_names = [str(r.get("School_Name", "")).strip().lower() for r in edu_history]
    degree_names = [str(r.get("Degree", "")).strip().lower() for r in edu_history]
    combined_signatures = [f"{s} - {d}" for s, d in zip(school_names, degree_names)]

    try:
        indices_to_drop = set()

        def _similarity(left: str, right: str) -> float:
            left = " ".join(left.lower().split())
            right = " ".join(right.lower().split())
            if not left or not right:
                return 0.0
            if left == right:
                return 1.0
            return SequenceMatcher(None, left, right).ratio()

        for i in range(n):
            if i in indices_to_drop:
                continue
            for j in range(i + 1, n):
                if j in indices_to_drop:
                    continue

                similarity_school = _similarity(school_names[i], school_names[j])
                similarity_degree = _similarity(degree_names[i], degree_names[j])

                school_match_exact = similarity_school >= 0.98
                degree_match_exact = similarity_degree >= 0.98

                is_duplicate = (
                        (similarity_school >= SIMILARITY_THRESHOLD and similarity_degree >= SIMILARITY_THRESHOLD) or
                        (school_match_exact and similarity_degree >= 0.6) or
                        (degree_match_exact and similarity_school >= 0.6)
                )

                if is_duplicate:
                    if len(degree_names[j]) > len(degree_names[i]):
                        indices_to_drop.add(i)
                        print(f"[EduDedupe] DROP A: {combined_signatures[i]!r} in favour of {combined_signatures[j]!r}")
                        break
                    else:
                        indices_to_drop.add(j)
                        print(f"[EduDedupe] DROP B: {combined_signatures[j]!r} in favour of {combined_signatures[i]!r}")

        if indices_to_drop:
            dropped = [edu_history[idx] for idx in indices_to_drop]
            print(f"[EduDedupe] Dropped {len(dropped)} duplicate(s): {dropped}")

        return [record for idx, record in enumerate(edu_history) if idx not in indices_to_drop]

    except Exception as local_err:
        print(f"⚠️ EduDedupe exception: {local_err}")
        return edu_history