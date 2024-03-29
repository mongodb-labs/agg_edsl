import json

import pymongo

from mongodsl.agg import aggregate

mongo = pymongo.MongoClient()
db = mongo.get_database("dsltest")

db.orders.drop()

db.orders.insert_many([{"k": x, "v": x % 2} for x in range(20)])


example = [
    {"$match": {"k": {"$gte": 10}}},
    {"$group": {"_id": "$v", "ks": {"$push": "$k"}}},
    {"$count": "count"},
]

print("expected:")
print(json.dumps(example, indent=4))
res = list(db.orders.aggregate(example))
print(res)


@aggregate
def matchGE10():
    match(k >= 10)


@aggregate
def groupAndCount():
    with group(v):
        ks = push(k)
    count("count")


test = matchGE10() | groupAndCount()

print("test:")
print(json.dumps(test.to_json(), indent=4))
res = list(test.apply(db.orders))
print(res)
