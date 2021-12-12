class CommandHandler:
    def __init__(self, command):
        self.command = command

    def handle(self, args):
        self.command.execute(args)
