from .base import CommandChain, ProcessDataCommandBase

class ProcessThermalDataCommand(ProcessDataCommandBase):
    factory_name = 'thermal'

cmd1 = ProcessThermalDataCommand()

# cmd1.set_next()

chain = CommandChain(head=cmd1, loop=False, max_loops=1)


