class HDFC_bank:
    def __init__(self, balance):
        self.__amount_limit = balance
        self.__trans_limit = 3
        self.__trans_done = 1

    def withdraw(self, amount):
        if self.__trans_done > self.__trans_limit:
            raise ValueError("transaction limit reached")
        if amount > self.__amount_limit:
            raise ValueError("Amount not under limit")
        print("collect the amount")
        self.__amount_limit -= amount
        print(f"remaining amount: {self.__amount_limit}")
        self.__trans_done += 1

    def deposit(self, amount):
        self.__amount_limit += amount
        print(f"New Limit: {self.__amount_limit}")


class Axis:
    def __init__(self, balance):
        self.__amount_limit = balance
        self.__trans_limit = 5
        self.__trans_done = 1

    def withdraw(self, amount):
        if self.__trans_done > self.__trans_limit:
            raise ValueError("transaction limit reached")
        if amount > self.__amount_limit:
            raise ValueError("Amount not under limit")
        print("collect the amount")
        self.__amount_limit -= amount
        print(f"remaining amount: {self.__amount_limit}")
        self.__trans_done += 1

    def deposit(self, amount):
        self.__amount_limit += amount
        print(f"New Limit: {self.__amount_limit}")
