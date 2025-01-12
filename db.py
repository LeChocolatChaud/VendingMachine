import json

with open("db.json") as f:
    data = json.load(f)

def check_person(name):
    return name in data["accounts"]

def check_coupon(code):
    return code in data["coupons"]

def get_account_money(name):
    return data["accounts"][name]["money"]

def set_account_money(name, money):
    data["accounts"][name]["money"] = money
    with open("db.json", "w") as f:
        json.dump(data, f, indent=4)