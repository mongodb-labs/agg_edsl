import json

import pymongo

from mongodsl.agg import aggregate, DEBUG

mongo = pymongo.MongoClient()
db = mongo.get_database("dsltest")

db.orders.drop()

db.orders.insert_many(
    [
        {
            "codename": "Ice Lake",
            "release_year": 2019,
            "products": [
                {
                    "model": "i7-1065G7",
                    "cores": 4,
                },
                {
                    "model": "i7-1060G7",
                    "cores": 4,
                },
            ],
        },
        {
            "codename": "Comet Lake",
            "release_year": 2020,
            "products": [
                {
                    "model": "i7-10750H",
                    "cores": 6,
                },
                {
                    "model": "i7-10610U",
                    "cores": 4,
                },
                {
                    "model": "i9-10910",
                    "cores": 10,
                },
                {
                    "model": "i9-10900KF",
                    "cores": 10,
                },
                {
                    "model": "i7-10700K",
                    "cores": 8,
                },
            ],
        },
        {
            "codename": "Tiger Lake",
            "release_year": 2021,
            "products": [
                {"model": "i9-11980HK", "cores": 8},
                {"model": "i7-11850HE", "cores": 8},
                {"model": "i7-11600H", "cores": 6},
            ],
        },
        {
            "codename": "Rocket Lake",
            "release_year": 2021,
            "products": [
                {"model": "i9-11900K", "cores": 8},
                {"model": "i7-11700K", "cores": 8},
                {"model": "i7-11600K", "cores": 6},
            ],
        },
        {
            "codename": "Alder Lake",
            "release_year": 2022,
            "products": [
                {"model": "i9-12900KF", "cores": 8},
                {"model": "i9-12900H", "cores": 6},
                {"model": "i7-1270P", "cores": 4},
                {"model": "i7-1265U", "cores": 2},
            ],
        },
    ]
)

@aggregate
def new_prod():
    match(release_year >= 2020)

@aggregate
def group_model():
    unwind(products)
    with group(products.cores):
        models = addToSet(products.model)
    cores = _id

@aggregate
def rm_id():
    del _id

pipe = new_prod() | group_model() | rm_id()
debug_pipe = pipe & DEBUG(mongo.get_database("debug_temp"))


print("test")
print(json.dumps(pipe.to_json(), indent=4))
res = list(pipe(db.orders))
print(res)

res = list(debug_pipe(db.orders))
for name, part in res:
    print(f"Output from: {name}")
    for p in part:
        print(f">>>> {p}")
