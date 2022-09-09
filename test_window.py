import pymongo as pm
from datetime import datetime
from mongodsl.agg import aggregate

mongo = pm.MongoClient()
db = mongo.get_database("stream_test")

print("Preparing data")

db.test_ob.drop()
db.test_ob.insert_many([{"k": x, "v": x % 2 == 0} for x in range(20)])


pipeline = [
    {"$window": {"type": "sliding", "unit": "event", "size": 5, "gap": 3}},
    {"$ob": {"mode": 0, "name": "window signal"}},
    {
        "$groups": {
            "_id": None,
            "res": {"$push": "$k"},
        }
    },
]

windows = [[]]

for r in db.test_ob.aggregate(pipeline):
    print(f"{r = }")


@aggregate
def sliding_window_test():
    window(type="sliding", unit="event", size=5, gap=3)
    ob(mode=0, name="window signal")
    with groups(None):
        res = push(k)


print(sliding_window_test().to_json())
for r in sliding_window_test()(db.test_ob):
    print(f"{r = }")
