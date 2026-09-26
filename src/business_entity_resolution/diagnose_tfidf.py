from pathlib import Path

import pandas as pd

from .normalization import normalize_record
from .retrieval import CharacterTfidfRetriever


DEV = Path("data/dev/train")


def load(path):
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )
    return [
        normalize_record(row)
        for row in df.to_dict(orient="records")
    ]


def main():
    print("Loading data...")

    s1 = load(DEV / "train_source1.tsv")
    s3 = load(DEV / "train_source3.tsv")

    s1_lookup = {r.entity_id: r for r in s1}

    target_ids = {
        "S3-38361126",
        "S3-96801034",
    }

    source1_ids = {
        "S1-100321232",
        "S1-101350071",
    }

    print("Building S3 name TF-IDF index...")

    retriever = CharacterTfidfRetriever(
        s3,
        field="name",
        ngram_range=(3, 5),
        top_k=len(s3),
        min_score=0.0,
    )

    print("Querying the two S1 records...")

    for s1_id in source1_ids:

        record = s1_lookup[s1_id]

        print("\n" + "=" * 80)
        print("S1:", s1_id)
        print("Name:", record.name_unicode)
        print("=" * 80)

        results = retriever.retrieve(record)

        found = None

        for result in results:
            if result.candidate_entity_id in target_ids:
                found = result
                break

        if found:
            print("TRUE TARGET FOUND")
            print("Candidate:", found.candidate_entity_id)
            print("Rank:", found.rank)
            print("Score:", found.score)
        else:
            print("TRUE TARGET NOT FOUND")

        print("\nTop 10 results:")

        for result in results[:10]:
            print(
                f"{result.rank:4d} | "
                f"{result.score:.6f} | "
                f"{result.candidate_entity_id}"
            )


if __name__ == "__main__":
    main()