from typing import Optional
from processors.factory_maker import make_factory_objects
from processors.types import ProcessorKinds


class CommandContext:
    def __init__(self, **kwargs):
        self.data = dict(kwargs)

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, value):
        self.data[key] = value


class CommandBase:
    def __init__(self):
        self.next_command: Optional['CommandBase'] = None

    def set_next(self, command: 'CommandBase') -> 'CommandBase':
        self.next_command = command
        return command 

    def execute(self, context: CommandContext):
        raise NotImplementedError("Subclasses must implement the execute method.")

class ProcessDataCommandBase(CommandBase):
    factory_name: ProcessorKinds

    def execute(self, context):
        return make_factory_objects(self.factory_name).process()



class CommandChain:
    def __init__(self, head: CommandBase, loop: bool = False, max_loops: Optional[int] = 1):
        self.head = head
        self.loop = loop
        self.max_loops = max_loops

    def run(self, context: Optional[CommandContext] = None):
        context = context or CommandContext()
        loop_count = 0

        while True:
            current:Optional['CommandBase'] = self.head
            while current:
                current.execute(context)
                current = current.next_command

            loop_count += 1
            if not self.loop:
                break
            if self.max_loops is not None and loop_count >= self.max_loops:
                break


