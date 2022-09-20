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
def matchGE10(factor):
    match(k > f * factor)


@aggregate
def removeId():
    del _id


@aggregate
def matchAndRemove(f):
    matchGE10(f)
    removeId()


print("embedded 2:")
print(json.dumps(matchAndRemove(2).to_json(), indent=4))
res = list(matchAndRemove(2)(db.orders))
print(res)

print("embedded 1.5:")
print(json.dumps(matchAndRemove(1.5).to_json(), indent=4))
res = list(matchAndRemove(1.5)(db.orders))
print(res)
