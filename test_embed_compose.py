import json

import pymongo

from mongodsl.agg import aggregate

mongo = pymongo.MongoClient()
db = mongo.get_database("dsltest")

db.orders.drop()

db.orders.insert_many([{"k": x, "v": x % 2, "f": x // 2} for x in range(20)])


example = [
    {"$match": {"$expr": {"$gt": ["$k", {"$multiply": ["$f", 2]}]}}},
    {"$unset": "_id"},
]

print("expected:")
print(json.dumps(example, indent=4))
res = list(db.orders.aggregate(example))
print(res)


@aggregate
def matchGE10():
    match(k > f * 2)


@aggregate
def removeId():
    del _id


@aggregate
def matchAndRemove():
    matchGE10()
    removeId()


print("embedded:")
print(json.dumps(matchAndRemove().to_json(), indent=4))
res = list(matchAndRemove()(db.orders))
print(res)
