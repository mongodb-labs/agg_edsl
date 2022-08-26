import json

import pymongo

from mongodsl.agg import aggregate

mongo = pymongo.MongoClient()
db = mongo.get_database("dsltest")

db.orders.drop()

db.orders.insert_many(
    [
        {
            "product": "apple",
            "price": 10,
        },
        {
            "product": "orange",
            "price": 5,
        },
    ]
)


@aggregate
def test(discountRatio):
    discount_price = floor(price * discountRatio)


rat = 0.8
print("test 0.8:")
print(json.dumps(test(rat).to_json(), indent=4))
res = list(test(rat).apply(db.orders))
print(res)

rat = 0.5
print("test 0.5:")
print(json.dumps(test(rat).to_json(), indent=4))
res = list(test(rat).apply(db.orders))
print(res)
