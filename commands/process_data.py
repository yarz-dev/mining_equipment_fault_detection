from .base import CommandChain, ProcessDataCommandBase

class ProcessThermalDataCommand(ProcessDataCommandBase):
    factory_name = 'thermal'

class ProcessAcousticDataCommand(ProcessDataCommandBase):
    factory_name = 'acoustic'

cmd1 = ProcessThermalDataCommand()
cmd2 = ProcessAcousticDataCommand()

cmd1.set_next(cmd2)

chain = CommandChain(head=cmd1, loop=False, max_loops=1)


