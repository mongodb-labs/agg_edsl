import json

import pymongo

from mongodsl.agg import aggregate

# PUSH 10
# LOAD_FIELD DATA 0
# BINOP ADD
# RETURN
# NOP
# END OF INSTR
# DATA SECTION
# 0 "key"
# END OF DATA
code = "".join(map(chr, [2, 10, 6, 0, 4, 0, 1, 0, 127, 107, 101, 121, 0]))

print(f"{code = }")

mongo = pymongo.MongoClient()
db = mongo.get_database("dsltest")

db.orders.drop()

db.orders.insert_many([{"key": x, "v": x % 2} for x in range(20)])


res = db.orders.aggregate([{"$set": {"foobar": {"$udf": code}}}])

for r in res:
    print(r)


@aggregate
def test_udf():
    foobar = udf(addKey10)
    with group(v):
        keys = push(key)


def addKey10(doc):
    return 10 + doc.key


print(test_udf().to_json())
res = test_udf()(db.orders)

print("test out")
for r in res:
    print(r)
