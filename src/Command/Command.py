from abc import ABC, abstractmethod


class Command(ABC):
    def __init__(self, next):
        self.next = next

    def execute(self, args):
        if self != None:
            self.handle(args)

        if self.next != None:
            self.next.execute(args)

    @abstractmethod
    def handle(self, args):
        print("Base class")
