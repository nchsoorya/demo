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

        print(f"\n🔬 [PAIRWISE EVALUATION LOG] Running All Combinations:")
        print(
            f"{'Record A String Signature':<70} | {'Record B String Signature':<70} | {'Comb':<6} | {'Sch':<6} | {'Deg':<6} | {'Status'}")
        print("-" * 235)

        for i in range(n):
            if i in indices_to_drop:
                continue
            for j in range(i + 1, n):
                if j in indices_to_drop:
                    continue

                # --- Real Vector Slice Extractions ---
                similarity_combined = _similarity(combined_signatures[i], combined_signatures[j])
                similarity_school = _similarity(school_names[i], school_names[j])
                similarity_degree = _similarity(degree_names[i], degree_names[j])

                school_match_exact = similarity_school >= 0.98
                degree_match_exact = similarity_degree >= 0.98

                # Double-Cross Matching Gates
                is_duplicate = (
                        # similarity_combined >= SIMILARITY_THRESHOLD or
                        (similarity_school >= SIMILARITY_THRESHOLD and similarity_degree >= SIMILARITY_THRESHOLD) or
                        (school_match_exact and similarity_degree >= 0.6) or
                        (degree_match_exact and similarity_school >= 0.6)
                )

                status_text = "KEEP"

                if is_duplicate:
                    if len(degree_names[j]) > len(degree_names[i]):
                        indices_to_drop.add(i)
                        status_text = "DROP_A"

                        # Print the detailed single line before breaking the inner loop step
                        print(
                            f"{combined_signatures[i]:<70} | {combined_signatures[j]:<70} | {similarity_combined:.4f} | {similarity_school:.4f} | {similarity_degree:.4f} | {status_text}")
                        break
                    else:
                        indices_to_drop.add(j)
                        status_text = "DROP_B"

                # Print clean inline metric summaries for all evaluations
                print(
                    f"{combined_signatures[i]:<70} | {combined_signatures[j]:<70} | {similarity_combined:.4f} | {similarity_school:.4f} | {similarity_degree:.4f} | {status_text}")

        dropped_records = [record for idx, record in enumerate(edu_history) if idx in indices_to_drop]
        print("\n🗑️ Dropped education items:")
        for dropped in dropped_records:
            print(dropped)

        # Reconstruct final data
        cleaned_edu = [record for idx, record in enumerate(edu_history) if idx not in indices_to_drop]

        # print("\n\n\n\n\n\n\n\n\n\n\n")
        # print(cleaned_edu)
        # print("\n\n\n\n\n\n\n\n\n\n\n")

        return cleaned_edu

    except Exception as local_err:
        print(f"⚠️ Vector pairing exception caught: {local_err}")
        return edu_history
