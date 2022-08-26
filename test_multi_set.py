import json

import pymongo

from mongodsl.agg import aggregate

mongo = pymongo.MongoClient()
db = mongo.get_database("dsltest")

db.orders.drop()

db.orders.insert_many(
    [
        {
            "customer_id": "jane",
            "total": 5,
            "quantity": 2,
        },
        {
            "customer_id": "bob",
            "total": 100,
            "quantity": 3,
        },
    ]
)


example = [
    {
        "$set": {
            "avg_unit_price": {"$floor": {"$divide": ["$total", "$quantity"]}},
            "total_price": {"$multiply": ["$total", "$quantity"]},
        }
    }
]


@aggregate
def test():
    avg_unit_price = floor(total / quantity)
    total_price = total * quantity


print("expected:")
print(json.dumps(example, indent=4))
res = list(db.orders.aggregate(example))
print(res)

print("test:")
print(json.dumps(test().to_json(), indent=4))
res = list(test().apply(db.orders))
print(res)
