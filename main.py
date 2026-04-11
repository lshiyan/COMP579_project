from chatarena.chatarena.chameleon_arena import ChameleonArena
from chatarena.chatarena.arena import Arena
arena = Arena.from_config("chatarena/examples/chameleon.json")
arena.launch_cli(interactive=False, max_steps=20)
