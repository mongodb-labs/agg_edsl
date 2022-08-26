import json

import pymongo

from mongodsl.agg import aggregate

mongo = pymongo.MongoClient()
db = mongo.get_database("dsltest")

db.orders.drop()

db.orders.insert_many([{"k": x, "v": x % 2} for x in range(20)])


example = [
    {"$set": {"v20": {"$multiply": ["$v", 20]}}},
    {"$group": {"_id": "$v20", "ka": {"$push": "$k"}, "ks": {"$addToSet": "$k"}}},
]


@aggregate
def test():
    with group(v * 20):
        ka = push(k)
        ks = addToSet(k)


print("expected:")
print(json.dumps(example, indent=4))
res = list(db.orders.aggregate(example))
print(res)

print("test:")
print(json.dumps(test().to_json(), indent=4))
res = list(test().apply(db.orders))
print(res)
