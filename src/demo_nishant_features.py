from src.nishant_features import build_pair_features


examples = [
    {
        "name1": "Payne Enterprises",
        "address1": "3315 Fremont Street, Peoria, IL",
        "country1": "US",

        "name2": "Payne Ã‰nterprises",
        "address2": "3315 FREMONT ST, PEORIA, IL",
        "country2": "US",

        "source": "S2",
        "score": 0.92,
        "rank": 1,
    },

    {
        "name1": "Lumay Boral",
        "address1": "1056 Belden Avenue, Akron, OH",
        "country1": "US",

        "name2": "Lumay BÃ³ral Inc.",
        "address2": "1056-1060 BELDEN AVE, PO BOX 8807, AKRON, OH",
        "country2": "US",

        "source": "S2",
        "score": 0.88,
        "rank": 1,
    },

    {
        "name1": "Dahlia Power Reliable Scientific LLC",
        "address1": "630 45th Terrace, Kansas City, MO",
        "country1": "US",

        "name2": "Dahlia Power Reliable",
        "address2": "KANSAS CITY, MO, 630 45ND TERRACE",
        "country2": "US",

        "source": "S2",
        "score": 0.85,
        "rank": 2,
    },

    {
        "name1": "Hendricks and Flowers Inc",
        "address1": "33 Sleepy Hollow Drive, Danbury, CT",
        "country1": "US",

        "name2": "Hendricks and Flowers Inc",
        "address2": "CT, SLEEPY HOLLOW DRIVE, DANBURY",
        "country2": "US",

        "source": "S2",
        "score": 0.91,
        "rank": 1,
    },
]


for i, example in enumerate(examples, start=1):

    print("\n" + "=" * 70)
    print(f"EXAMPLE {i}")
    print("=" * 70)

    features = build_pair_features(
        name1=example["name1"],
        address1=example["address1"],
        country1=example["country1"],
        name2=example["name2"],
        address2=example["address2"],
        country2=example["country2"],
        candidate_source=example["source"],
        retrieval_score=example["score"],
        retrieval_rank=example["rank"],
    )

    for key, value in features.items():
        print(f"{key:30} : {value}")
